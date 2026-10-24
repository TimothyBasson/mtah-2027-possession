"""Play the whole match with the tracking drawn on the video, a synced top-down pitch, and possession.

    python viz/combined/make_combined.py --open
    python viz/combined/make_combined.py --pred evaluation/results/baseline_smooth50.csv --start 1:20:05 --open

One HTML page for the entire tactical video, with three views driven by one clock -- the video:

- On the video: every tracked slot and the ball, projected onto the picture through that frame's
  homography. Where a ring sits off a player, or a player has no ring, the tracking is wrong there.
- Beside it: the same frame as a top-down pitch, as in viz/replay.
- Under it: possession for the whole match and, zoomed in, for the minute either side of the
  playhead. Truth on top, where the public ground truth has any; your file below. Click to seek.

The tracking for 199k frames is too big for one page, so it is written once as one-minute chunk
files under data/viz/match/, shared by every page, and the page loads the chunks around the
playhead as it plays. Each page itself holds only the labels. Chunks are loaded with <script> tags,
so the page works opened from disk as well as served (see .claude/launch.json).

Standard library only, and the video plays straight from disk like viz/overlay.

The output contains match data
------------------------------
The chunks hold tracking positions, so everything is written under data/, which .gitignore keeps
out of git. Do not commit it or share it: the data terms cover derived data too.
"""

import argparse
import csv
import json
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "viz", "replay"))
from games import DATA_DIR, require, source_frames  # noqa: E402
from make_replay import FPS, PUBLIC_GT, clock_spec, parse_time, read_labels, read_teams  # noqa: E402

# Params
TEMPLATE = os.path.join(HERE, "combined_template.html")
OUT_DIR = os.path.join(DATA_DIR, "viz")
CHUNK_DIR = os.path.join(OUT_DIR, "match")
CHUNK = 60 * FPS                 # frames per chunk file: one minute, ~370 KB
DEFAULT_START = "7:27.4"         # start of the public ground truth
SERVE_URL = "http://localhost:8765"   # where .claude/launch.json's "viz" config serves data/
HOM_KEYS = ("h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")


def chunk_path(k):
    return os.path.join(CHUNK_DIR, "chunk_{:04d}.js".format(k))


def write_chunks(n_frames):
    """Stream the ATD and the homography together into one-minute chunk files.

    Each chunk is a script calling MATCH_CHUNK(k, {slots, ball, H}) with one entry per frame, in
    the same encoding as viz/replay: positions in thousandths of the pitch, a slot as
    [slot, x, y, estimated], the ball or null, the homography's nine values or null where invalid.
    """
    raw_path, hom_path = require(["atd_raw", "homography"])
    os.makedirs(CHUNK_DIR, exist_ok=True)
    chunk = {"slots": [], "ball": [], "H": []}
    with open(raw_path) as atd, open(hom_path, newline="") as hom_handle:
        hom = csv.DictReader(hom_handle)
        for frame in range(n_frames):
            frame_id, players, ball = next(atd).rstrip("\n").split(":")
            row = next(hom)
            if int(frame_id) - 1 != frame or int(row["frame"]) != frame:
                raise SystemExit("tracking or homography out of step at video frame {}".format(frame))
            slots = []
            for slot, group in enumerate(players.split(";"), start=1):
                x, y, speed = group.split(",")
                if x != "NaN":
                    slots.append([slot, round(float(x) * 1000), round(float(y) * 1000),
                                  1 if speed.strip() == "NaN" else 0])
            bx, by = ball.split(",")[:2]
            chunk["slots"].append(slots)
            chunk["ball"].append(None if bx == "NaN" else [round(float(bx) * 1000),
                                                           round(float(by) * 1000)])
            chunk["H"].append([float("%.6g" % float(row[k])) for k in HOM_KEYS]
                              if row["valid"].strip() == "1" else None)
            if frame % CHUNK == CHUNK - 1 or frame == n_frames - 1:
                with open(chunk_path(frame // CHUNK), "w") as out:
                    out.write("MATCH_CHUNK({},{});\n".format(
                        frame // CHUNK, json.dumps(chunk, separators=(",", ":"))))
                chunk = {"slots": [], "ball": [], "H": []}


def label_string(path, n_frames):
    """'XXHHHAAD...' -- one character per video frame, X where the file has no row."""
    labels = read_labels(path, 0, n_frames - 1)
    return "".join(labels.get(frame, "X") for frame in range(n_frames))


def build(pred_path=None, gt_path=PUBLIC_GT):
    n_frames = source_frames()
    slots, names = read_teams(require(["atd_meta"])[0])
    return {"teams": slots, "names": names, "n_frames": n_frames, "chunk": CHUNK,
            "clock": clock_spec(),
            "truth": label_string(gt_path, n_frames) if gt_path else None,
            "pred": label_string(pred_path, n_frames) if pred_path else None,
            "pred_name": os.path.basename(pred_path) if pred_path else None}


def write_page(payload, out_path, video_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    rel = lambda path: os.path.relpath(path, out_dir).replace(os.sep, "/")
    payload = dict(payload, video=rel(video_path), chunks=rel(CHUNK_DIR))
    with open(TEMPLATE) as handle:
        template = handle.read()
    # "</" inside a <script> would end it early; JSON allows the escaped form.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    with open(out_path, "w") as handle:
        handle.write(template.replace("__COMBINED_DATA__", data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=parse_time, default=parse_time(DEFAULT_START),
                        help="tactical video time the page opens at: 7:27.4, 1:14:10, 453.4 or "
                             "f11185 (default: start of the public ground truth)")
    parser.add_argument("--pred", help="a frame,label submission CSV to show alongside the truth")
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground truth CSV (default: the public 5 minutes); 'none' to hide")
    parser.add_argument("--out", help="where to write the page (default: data/viz/...)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rewrite the tracking chunks even if they already exist")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    args = parser.parse_args()

    gt = None if args.gt.lower() == "none" else args.gt
    n_frames = source_frames()
    last_chunk = chunk_path((n_frames - 1) // CHUNK)
    if args.rebuild or not os.path.exists(last_chunk):
        print("writing tracking chunks to {} (once; takes a minute)".format(CHUNK_DIR))
        write_chunks(n_frames)

    name = "match" + ("_" + os.path.splitext(os.path.basename(args.pred))[0] if args.pred else "")
    out = args.out or os.path.join(OUT_DIR, name + ".html")
    write_page(build(args.pred, gt), out, require(["video"])[0])

    start = "#f{}".format(min(n_frames - 1, int(round(args.start * FPS))))
    print("wrote {}".format(out))
    if os.path.dirname(os.path.abspath(out)) == os.path.abspath(OUT_DIR):
        print("served (the 'viz' preview in .claude/launch.json): {}/viz/{}{}".format(
            SERVE_URL, os.path.basename(out), start))
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out) + start)
