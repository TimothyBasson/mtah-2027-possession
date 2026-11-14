"""Play the whole match with the tracking drawn on the video, a synced top-down pitch, and possession.

    python viz/combined/make_combined.py --open
    python viz/combined/make_combined.py --pred evaluation/results/baseline_smooth50.csv --start 1:20:05 --open
    uv run python viz/combined/make_combined.py --match all --open
    uv run python viz/combined/make_combined.py --match all --explain --open
    uv run python viz/combined/make_combined.py --deep --open

--explain shows possession/'s labels with how it reached them, clue by clue, under the strips.
--deep shows the learned model (deep/) at work, stage by stage: what it reads from the players, its
score for every frame, where it guesses the ball is, each smoothing step and the possessions out.

--match plays an outside match from data/external/ (external/README.md) instead, or with 'all', writes
a page for BAR-ATH and every outside match with a dropdown to switch between them. Those matches
have no video, so a timer drives the page and the pitch takes the video's place. Needs numpy.

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
import base64
import csv
import glob
import json
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "viz", "replay"))
from games import DATA_DIR, GAME, GAMES, require, source_frames, video_path  # noqa: E402
from make_replay import FPS, PUBLIC_GT, clock_spec, parse_time, read_labels, read_teams  # noqa: E402

# Params
TEMPLATE = os.path.join(HERE, "combined_template.html")
OUT_DIR = os.path.join(DATA_DIR, "viz")
CHUNK_DIR = os.path.join(OUT_DIR, "match")
CHUNK = 60 * FPS                 # frames per chunk file: one minute, ~370 KB
DEFAULT_START = "7:27.4"         # start of the public ground truth
SERVE_URL = "http://localhost:8765"   # where .claude/launch.json's "viz" config serves data/
HOM_KEYS = ("h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")


def chunk_path(k, chunk_dir=CHUNK_DIR):
    return os.path.join(chunk_dir, "chunk_{:04d}.js".format(k))


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
            "clock": clock_spec(), "pitch": list(GAMES[GAME]["pitch"]),
            "truth": label_string(gt_path, n_frames) if gt_path else None,
            "pred": label_string(pred_path, n_frames) if pred_path else None,
            "pred_name": os.path.basename(pred_path) if pred_path else None}


def build_external(match, pred_path=None, rebuild=False):
    """The page payload for an outside match, writing its chunks to data/viz/match_<match>/ first if
    needed. Same chunk format as write_chunks, without a homography; frames are npz rows."""
    import numpy as np
    path = os.path.join(DATA_DIR, "external", match + ".npz")
    if not os.path.exists(path):
        raise SystemExit("no {} -- build it with external/build.py".format(path))
    z = np.load(path)
    chunk_dir = os.path.join(OUT_DIR, "match_" + match)
    labels = z["labels"]
    n_frames, size = len(labels), z["pitch"]
    if rebuild or not os.path.exists(chunk_path((n_frames - 1) // CHUNK, chunk_dir)):
        print("writing {} tracking chunks to {}".format(match, chunk_dir))
        os.makedirs(chunk_dir, exist_ok=True)
        players = np.rint(z["players"][..., :2] / size * 1000)
        ball = np.rint(z["ball"][:, :2] / size * 1000)
        for k in range(0, n_frames, CHUNK):
            slots = [[[s + 1, int(x), int(y), 0] for s, (x, y) in enumerate(frame) if x == x]
                     for frame in players[k:k + CHUNK].tolist()]
            balls = [[int(x), int(y)] if x == x else None for x, y in ball[k:k + CHUNK].tolist()]
            with open(chunk_path(k // CHUNK, chunk_dir), "w") as out:
                out.write("MATCH_CHUNK({},{});\n".format(k // CHUNK, json.dumps(
                    {"slots": slots, "ball": balls, "H": None}, separators=(",", ":"))))
    halves = z["halves"].tolist()[:2]
    return {"teams": {s + 1: "HA"[t] for s, t in enumerate(z["team"].tolist())},
            "names": {"H": "Home", "A": "Away"}, "match": match, "n_frames": n_frames, "chunk": CHUNK,
            "pitch": size.tolist(),
            "clock": {"fps": int(z["fps"]), "kickoff": [h[0] for h in halves], "end": [h[1] for h in halves]},
            "truth": "".join(labels.tolist()),
            "pred": label_string(pred_path, n_frames) if pred_path else None,
            "pred_name": os.path.basename(pred_path) if pred_path else None}, chunk_dir


def explain(match):
    """The possession model's output and working for a match (None = BAR-ATH), for the decomposition
    lanes: its labels, its frame-by-frame guess before smoothing, each clue's lean as log-odds in
    tenths (base64 int8), the ball as the model saw it, and scores where the whole match is labelled."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "possession"))
    import features
    import model
    import run
    F, labels = run.frames(match)
    x = model.explain(F)
    tenths = lambda v: base64.b64encode(np.clip(np.rint(np.nan_to_num(v) * 10), -127, 127)
                                        .astype(np.int8).tobytes()).decode()
    out = {"raw": "".join(x["raw"]), "lanes": [[name, kind, tenths(v)] for name, kind, v in x["lanes"]],
           "seen": "".join(np.where(F["seen"], "1", "0")),
           # per half: first, last frame, then +1/-1 for whether home's and away's own goal is at x=0
           "own_goal": [[int(a), int(b), h, w] for (a, b), h, w in zip(model.HALVES, *features.own_goal(
               F["players"], F["team"], model.HALVES))]}
    if x.get("dead"):  # the stoppage model's signals, in hundredths, int16, -32768 for none
        hundredths = lambda v: base64.b64encode(np.where(np.isfinite(v), np.clip(np.rint(np.nan_to_num(v) * 100),
                                                -32767, 32767), -32768).astype("<i2").tobytes()).decode()
        out["dead"] = {"signals": list(x["dead"]["signals"]), "fit": x["dead"]["fit"],
                       "x": [hundredths(v) for v in x["dead"]["x"].T]}
    if labels is not None:
        out["scores"] = scores(labels, x["pred"])
    return "".join(x["pred"]), out


def scores(truth, pred):
    """The headline metrics of pred against truth, both arrays of H / A / D / X per frame."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    truth, pred = np.asarray(truth), np.asarray(pred)
    spells = lambda l: int(1 + (l[l != "X"][1:] != l[l != "X"][:-1]).sum())
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)
    return [["spellF1", round(r["spell_f1_mean"], 3)], ["macroF1", round(r["macro_f1"], 3)],
            ["alive/dead F1", round(r["alive_dead_f1"], 3)],
            ["team acc", round(r["team_acc_both_alive"], 3)],
            ["spells (model / truth)", "{} / {}".format(spells(pred[truth != "X"]), spells(truth))]]


def deep(match, truth, tag):
    """The learned model's labels for a match (None = BAR-ATH) and what it produced at each stage,
    from deep/explain.py's file, which is computed here if it is missing (that needs torch).

    Per half, at the model's 5 fps and base64: the H, A, D probabilities (uint8, 250 = 1), where the
    ball head puts the ball (int16, thousandths of the pitch), the chance of a change of label
    within 0.4 s (uint8, 250 = 1) and how far the ball guess is from the tracked ball (uint8,
    0.2 m steps, 255 where the ball is not tracked). Per video frame: the label before smoothing
    and after the switching cost; the label after short runs are absorbed is the page's estimate."""
    import numpy as np
    path = os.path.join(ROOT, "data", "cache", "deep", tag, "viz", (match or "bar_ath") + ".npz")
    if os.path.exists(path):
        z = np.load(path)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("deep_explain", os.path.join(ROOT, "deep", "explain.py"))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as error:
            raise SystemExit("no {} and it can't be computed here ({}). Run deep/explain.py where torch "
                             "is installed and copy the file over.".format(path, error))
        z = module.explain(match, tag)
    b64 = lambda a, kind: base64.b64encode(np.ascontiguousarray(a).astype(kind).tobytes()).decode()
    n_halves = len(z["halves"])
    pitch = np.array(GAMES[GAME]["pitch"], np.float32)
    if match:
        pitch = np.load(os.path.join(DATA_DIR, "external", match + ".npz"))["pitch"].astype(np.float32)
    err = np.concatenate([z["err%d" % i] for i in range(n_halves)])
    live = np.concatenate([z["p%d" % i][:, 2] for i in range(n_halves)]) < 0.5
    seen = np.isfinite(err)
    cost, min_team, min_dead = (float(v) for v in z["decode"])
    out = {"tag": tag, "step": int(z["step"]), "reach": int(z["reach"]),
           "decode": {"cost": cost, "min_team": min_team, "min_dead": min_dead},
           "halves": [[int(a), int(b)] for a, b in z["halves"]],
           "p": [b64(np.rint(z["p%d" % i] * 250), np.uint8) for i in range(n_halves)],
           "ball": [b64(np.clip(np.rint(z["ball%d" % i] / pitch * 1000), -32000, 32000), "<i2")
                    for i in range(n_halves)],
           "edge": [b64(np.rint(z["edge%d" % i] * 250), np.uint8) for i in range(n_halves)],
           "err": [b64(np.where(np.isfinite(z["err%d" % i]), np.clip(np.rint(np.nan_to_num(z["err%d" % i]) * 5), 0, 254), 255), np.uint8)
                   for i in range(n_halves)],
           "raw": "".join(z["raw"]), "switch": "".join(z["switch"]),
           # the ball head only learned from live play, so that is where its error means something
           "ball_err": round(float(np.median(err[seen & live])), 1) if (seen & live).any() else None}
    if truth and any(c in truth for c in "HAD"):
        out["scores"] = scores(list(truth), z["final"])
    return "".join(z["final"]), out


def write_page(payload, out_path, video_path, chunk_dir=CHUNK_DIR):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    rel = lambda path: os.path.relpath(path, out_dir).replace(os.sep, "/")
    payload = dict(payload, video=video_path and rel(video_path), chunks=rel(chunk_dir))
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
    parser.add_argument("--match", help="an outside match in data/external/ (idsse_J03WMX, metrica_1, ...), "
                        "or 'all' for BAR-ATH and every outside match with a switcher (needs numpy)")
    parser.add_argument("--pred", help="a frame,label submission CSV to show alongside the truth")
    parser.add_argument("--explain", action="store_true",
                        help="show possession/'s output with its working: each clue's lean, the guess "
                             "before smoothing, wrong frames and, on outside matches, the scores")
    parser.add_argument("--deep", nargs="?", const="v2", metavar="TAG",
                        help="show the learned model in deep/ at work, stage by stage: its score for "
                             "every frame, its ball guess, each smoothing step and the possessions out "
                             "(TAG is the model version, default v2)")
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground truth CSV (default: the public 5 minutes); 'none' to hide")
    parser.add_argument("--out", help="where to write the page (default: data/viz/...)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rewrite the tracking chunks even if they already exist")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    args = parser.parse_args()

    gt = None if args.gt.lower() == "none" else args.gt
    if (args.explain or args.deep) and args.pred:
        parser.error("--explain and --deep show the model's own labels; drop --pred")
    if args.explain and args.deep:
        parser.error("--explain is possession/'s working and --deep the learned model's: one page each")
    matches = [args.match]
    if args.match == "all":
        if args.pred or args.out:
            parser.error("--match all can't take --pred or --out: frames and paths differ per game")
        matches = [None] + sorted(os.path.basename(f)[:-4] for f in
                                  glob.glob(os.path.join(DATA_DIR, "external", "*.npz")))
    suffix = "_" + os.path.splitext(os.path.basename(args.pred))[0] if args.pred else "_explain" if args.explain else "_deep" if args.deep else ""
    outs = [args.out or os.path.join(OUT_DIR, "match" + ("_" + m if m else "") + suffix + ".html")
            for m in matches]
    games = [[m or "BAR-ATH", os.path.basename(o)] for m, o in zip(matches, outs)]

    for match, out in zip(matches, outs):
        if match:
            payload, chunk_dir = build_external(match, args.pred, args.rebuild)
            video = None
        else:
            n_frames = source_frames()
            if args.rebuild or not os.path.exists(chunk_path((n_frames - 1) // CHUNK)):
                print("writing tracking chunks to {} (once; takes a minute)".format(CHUNK_DIR))
                write_chunks(n_frames)
            payload, chunk_dir, video = build(args.pred, gt), CHUNK_DIR, video_path()
            if not os.path.isfile(video):
                print("note: no video at {}; the page will look for it there".format(video))
        if args.explain:
            payload["pred"], payload["explain"] = explain(match)
            payload["pred_name"] = "possession model"
        if args.deep:
            payload["pred"], payload["deep"] = deep(match, payload["truth"], args.deep)
            payload["pred_name"] = "learned model " + args.deep
        if len(games) > 1:
            payload["games"] = games
        write_page(payload, out, video, chunk_dir)
        print("wrote {}".format(out))

    n_frames = payload["n_frames"] if len(matches) == 1 else source_frames()
    # BAR-ATH opens at its default start; outside matches at the time given, else their kickoff.
    start = "#f{}".format(min(n_frames - 1, int(round(args.start * FPS))))
    if matches[0] and args.start == parse_time(DEFAULT_START):
        start = ""
    out = outs[0]
    if os.path.dirname(os.path.abspath(out)) == os.path.abspath(OUT_DIR):
        print("served (the 'viz' preview in .claude/launch.json): {}/viz/{}{}".format(
            SERVE_URL, os.path.basename(out), start))
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out) + start)
