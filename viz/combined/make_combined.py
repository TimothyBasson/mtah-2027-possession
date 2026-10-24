"""Play the tactical video with the tracking drawn on it, a synced top-down pitch, and possession.

    python viz/combined/make_combined.py --start 7:27.4 --duration 60 --open
    python viz/combined/make_combined.py --start 7:27.4 --pred evaluation/results/baseline_smooth50.csv --open

One HTML page, three views driven by one clock -- the video:

- On the video: every tracked slot and the ball, projected onto the picture through that frame's
  homography. Where a ring sits off a player, or a player has no ring, the tracking is wrong there.
- Beside it: the same frame as a top-down pitch, as in viz/replay.
- Under it: possession across the clip, truth on top and your file below. Click it to seek.

It reuses viz/replay's readers, so it is standard library only, and it plays the video straight
from disk like viz/overlay. Frame-exact (no sampling), which is why clips are capped at 10 minutes (a few MB of page).

The output contains match data
------------------------------
The page embeds tracking positions, so it is written under data/ by default, which .gitignore
keeps out of git. Do not commit it or share it: the data terms cover derived data too.
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
from make_replay import (FPS, PUBLIC_GT, clock_spec, parse_time, read_labels,  # noqa: E402
                         read_teams, read_tracking)

# Params
TEMPLATE = os.path.join(HERE, "combined_template.html")
OUT_DIR = os.path.join(DATA_DIR, "viz")
DEFAULT_START = "7:27.4"         # start of the public ground truth
DEFAULT_DURATION = 60.0
MAX_DURATION = 10 * 60           # every frame is kept, ~4 MB of page at the cap
SERVE_URL = "http://localhost:8765"   # where .claude/launch.json's "viz" config serves data/


def read_homography(path, first, last):
    """[[h00..h22] or None, ...] for frames first..last; None where the frame has no valid matrix."""
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame < first:
                continue
            if frame > last:
                break
            if row["valid"].strip() != "1":
                out.append(None)
                continue
            out.append([float("%.6g" % float(row[k])) for k in
                        ("h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")])
    return out


def label_string(labels, first, last):
    """'HHHAAD...' -- one character per frame of the clip, X where the file has no row."""
    return "".join(labels.get(frame, "X") for frame in range(first, last + 1))


def build(start_s, duration_s, pred_path=None, gt_path=PUBLIC_GT):
    n_frames = source_frames()
    first = max(0, int(round(start_s * FPS)))
    last = min(n_frames - 1, first + int(round(duration_s * FPS)) - 1)
    if first >= n_frames:
        raise SystemExit("start {:.1f} s is past the end of the video ({:.1f} s)"
                         .format(start_s, n_frames / float(FPS)))

    meta_path, raw_path, hom_path = require(["atd_meta", "atd_raw", "homography"])
    slots, names = read_teams(meta_path)
    frames = read_tracking(raw_path, first, last, step=1)
    homography = read_homography(hom_path, first, last)
    if len(frames) != len(homography):
        raise SystemExit("tracking has {} frames and homography {} for {}..{}"
                         .format(len(frames), len(homography), first, last))

    return {"teams": slots, "names": names, "first": first, "last": last, "clock": clock_spec(),
            "slots": [f[1] for f in frames], "ball": [f[2] for f in frames], "H": homography,
            "truth": label_string(read_labels(gt_path, first, last), first, last) if gt_path else None,
            "pred": label_string(read_labels(pred_path, first, last), first, last) if pred_path else None,
            "pred_name": os.path.basename(pred_path) if pred_path else None}


def default_out(start_s, duration_s, pred_path):
    hours, rest = divmod(start_s, 3600)
    name = "combined_{:d}-{:02d}-{:04.1f}_{:d}s".format(int(hours), int(rest // 60), rest % 60,
                                                         int(round(duration_s)))
    if pred_path:
        name += "_" + os.path.splitext(os.path.basename(pred_path))[0]
    return os.path.join(OUT_DIR, name + ".html")


def write_page(payload, out_path, video_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(payload, video=os.path.relpath(video_path, out_dir).replace(os.sep, "/"))
    with open(TEMPLATE) as handle:
        template = handle.read()
    # "</" inside a <script> would end it early; JSON allows the escaped form.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    with open(out_path, "w") as handle:
        handle.write(template.replace("__COMBINED_DATA__", data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=parse_time, default=parse_time(DEFAULT_START),
                        help="tactical video time to start at: 7:27.4, 1:14:10, 453.4 or f11185 "
                             "(default: start of the public ground truth)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="seconds of video to include (default %(default)s)")
    parser.add_argument("--pred", help="a frame,label submission CSV to show alongside the truth")
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground truth CSV (default: the public 5 minutes); 'none' to hide")
    parser.add_argument("--out", help="where to write the page (default: data/viz/...)")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    args = parser.parse_args()

    if not 0 < args.duration <= MAX_DURATION:
        parser.error("--duration must be between 0 and {} seconds".format(MAX_DURATION))
    gt = None if args.gt.lower() == "none" else args.gt

    payload = build(args.start, args.duration, args.pred, gt)
    out = args.out or default_out(args.start, args.duration, args.pred)
    write_page(payload, out, require(["video"])[0])

    print("wrote {}".format(out))
    print("video frames {}..{}  ({} frames)".format(payload["first"], payload["last"],
                                                     len(payload["slots"])))
    if os.path.dirname(os.path.abspath(out)) == os.path.abspath(OUT_DIR):
        print("served (the 'viz' preview in .claude/launch.json): {}/viz/{}".format(
            SERVE_URL, os.path.basename(out)))
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))
