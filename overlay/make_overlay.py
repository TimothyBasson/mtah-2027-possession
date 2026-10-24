"""Play the tactical video with your estimated possession and the ground truth drawn over it.

    python overlay/make_overlay.py --pred evaluation/results/baseline_smooth50.csv --open
    python overlay/make_overlay.py --open                      # ground truth only

Writes one HTML page that plays data/tactical/BAR-ATH_tactical.mp4 straight from disk -- nothing is
re-encoded -- and shows, for whatever frame is on screen, the label in your submission and the
true label where the public ground truth covers it (video 7:27.4 to 12:27.4). Seek, pause and
change speed with the normal video controls; the labels follow.

Standard library only. The page references the video by a relative path, so keep it where it was
written (data/viz/ by default) or regenerate it with --out. It holds labels but no tracking, and it
is written under data/, which .gitignore keeps out of git.
"""

import argparse
import csv
import json
import os
import sys
import webbrowser
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402

# Params
TEMPLATE = os.path.join(HERE, "overlay_template.html")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
OUT_DIR = os.path.join(DATA_DIR, "viz")
LABELS = ("H", "A", "D", "X")


def read_runs(path, n_frames):
    """A frame,label CSV as [[first_frame, label], ...] runs -- 199k rows become a few thousand."""
    labels = ["X"] * n_frames
    bad = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            frame, label = int(row["frame"]), row["label"].strip()
            if 0 <= frame < n_frames and label in LABELS:
                labels[frame] = label
            else:
                bad += 1
    if bad:
        print("warning: {} rows of {} were out of range or not one of {} -- check the file with "
              "evaluation/validate_submission.py".format(bad, path, "/".join(LABELS)))
    runs = []
    for frame, label in enumerate(labels):
        if not runs or runs[-1][1] != label:
            runs.append([frame, label])
    return runs


def team_names(meta_path):
    """{'H': home name, 'A': away name} from the ATD metadata XML."""
    root = ET.parse(meta_path).getroot()
    score = root.find(".//Score")
    side = {score.get("idLocalTeam"): "H", score.get("idVisitingTeam"): "A"}
    return {side[team.get("id")]: team.findtext("Name")
            for team in root.iter("Team") if team.get("id") in side}


def build(pred_path=None, gt_path=PUBLIC_GT):
    n_frames = source_frames()
    return {"fps": GAMES[GAME]["fps"], "n_frames": n_frames,
            "names": team_names(require(["atd_meta"])[0]),
            "truth": read_runs(gt_path, n_frames) if gt_path else None,
            "pred": read_runs(pred_path, n_frames) if pred_path else None,
            "pred_name": os.path.basename(pred_path) if pred_path else None}


def write_page(payload, out_path, video_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(payload, video=os.path.relpath(video_path, out_dir).replace(os.sep, "/"))
    with open(TEMPLATE) as handle:
        template = handle.read()
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    with open(out_path, "w") as handle:
        handle.write(template.replace("__OVERLAY_DATA__", data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", help="your frame,label submission CSV")
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground truth CSV (default: the public 5 minutes); 'none' to hide")
    parser.add_argument("--out", help="where to write the page (default: data/viz/...)")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    args = parser.parse_args()

    gt = None if args.gt.lower() == "none" else args.gt
    if not gt and not args.pred:
        parser.error("nothing to show: pass --pred, or keep the ground truth")

    video = require(["video"])[0]
    name = "overlay" + ("_" + os.path.splitext(os.path.basename(args.pred))[0] if args.pred else "")
    out = args.out or os.path.join(OUT_DIR, name + ".html")
    write_page(build(args.pred, gt), out, video)

    print("wrote {}".format(out))
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))
