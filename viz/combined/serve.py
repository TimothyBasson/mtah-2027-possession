"""Serve data/ for the match viewer, with the range requests the video needs and a notes store.

    python3 viz/combined/serve.py              # http://localhost:8765, as .claude/launch.json's "viz"

The same as `npx http-server data -p 8765 -c-1` for the pages, plus somewhere to keep the notes you
take while watching: the page reads them at GET /api/notes/<match> and adds, changes or removes
one at a time at PUT / DELETE /api/notes/<match>/<id>, so two open pages never write over each
other's notes. They land in
data/viz/notes/<match>.json, one file per match whichever prediction the page shows. data/ is
git-ignored, so notes stay on this machine like the rest of the derived match data.

Standard library only; listens on 127.0.0.1, since anyone who can reach it can write notes.
"""
import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
NOTES = os.path.join(DATA, "viz", "notes")
API = re.compile(r"^/api/notes/([^/]+)(?:/([^/]+))?$")
NOTE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LOCK = threading.Lock()
KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
MAX_BODY = 5 * 1024 * 1024
mimetypes.add_type("video/mp4", ".mp4")


def _read(path):
    if not os.path.isfile(path):
        return {"match": os.path.basename(path)[:-5], "notes": []}
    with open(path) as f:
        body = json.load(f)
    body.setdefault("notes", [])
    return body


def _write(path, body):
    os.makedirs(NOTES, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=NOTES, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(body, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)                          # never a half-written file


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, *args):
        pass

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _notes_target(self):
        """None if this isn't the notes API; else (file, note id or None), file '' for a bad name."""
        m = API.match(urlsplit(self.path).path)
        if not m:
            return None
        key, nid = unquote(m.group(1)), m.group(2) and unquote(m.group(2))
        ok = KEY.match(key) and (nid is None or NOTE_ID.match(nid))
        return (os.path.join(NOTES, key + ".json") if ok else ""), nid

    def _notes_reply(self, path):
        self._json(HTTPStatus.OK, _read(path))

    def do_GET(self):
        target = self._notes_target()
        if target is not None:
            if not target[0]:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "bad match name"})
            return self._notes_reply(target[0])
        if not self._range():
            super().do_GET()

    def do_PUT(self):
        """Add or replace one note, by its id. One note at a time, so two pages open on the same
        match (or a page left open with an old list) never write over each other's notes."""
        target = self._notes_target()
        if not target or not target[0] or not target[1]:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "PUT /api/notes/<match>/<note id>"})
        path, nid = target
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too big"})
        try:
            note = json.loads(self.rfile.read(n))
            if not isinstance(note, dict) or note.get("id") != nid or not isinstance(note.get("frame"), int):
                raise ValueError
        except ValueError:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "expected a note with this id and a frame"})
        with LOCK:
            body = _read(path)
            body["notes"] = sorted([x for x in body["notes"] if x.get("id") != nid] + [note], key=lambda x: x.get("frame", 0))
            _write(path, body)
        self._notes_reply(path)

    def do_DELETE(self):
        target = self._notes_target()
        if not target or not target[0] or not target[1]:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "DELETE /api/notes/<match>/<note id>"})
        path, nid = target
        with LOCK:
            body = _read(path)
            body["notes"] = [x for x in body["notes"] if x.get("id") != nid]
            _write(path, body)
        self._notes_reply(path)

    def _range(self):
        """Answer a 'Range: bytes=a-b' request for a file with 206 and just those bytes; False to
        leave the request to the plain handler. The video seeks with these."""
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = header and re.match(r"^bytes=(\d*)-(\d*)$", header.strip())
        if not m or not os.path.isfile(path) or m.group(1) + m.group(2) == "":
            return False
        size = os.path.getsize(path)
        if m.group(1) == "":
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", "bytes */{}".format(size))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left:
                    buf = f.read(min(left, 1 << 16))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass                               # the video seeked elsewhere and dropped this one
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=DATA))
    print("serving {} at http://localhost:{} (notes in {})".format(DATA, args.port, os.path.relpath(NOTES, ROOT)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
