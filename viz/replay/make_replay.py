"""Build an animated replay of the tracking for any stretch of the match, as one HTML page.

    python viz/replay/make_replay.py --start 7:33.4 --duration 40
    python viz/replay/make_replay.py --start 7:33.4 --pred evaluation/results/baseline_smooth50.csv
    python viz/replay/make_replay.py --start 1:20:05 --duration 60 --open
    uv run python viz/replay/make_replay.py --match idsse_J03WMX --start 10:00 --open
    uv run python viz/replay/make_replay.py --match all --start 10:00 --open

The page draws every tracked slot on a top-down pitch, with the ball, and plays it back. Alongside
it shows the tactical video time (seek your video player there to watch along), the game clock,
the true possession wherever the public ground truth covers the frame, and -- when you pass
--pred -- what your submission says at that moment.

Standard library only, on purpose: it reads the raw ATD by hand the same way starter/baseline.py
does, so it runs in any Python 3 without numpy, pandas or kloppy. The exception is --match, which
replays one of the outside matches in data/external/ (see external/README.md) and needs numpy.

What is on the page
-------------------
- A filled circle is a slot with a measured position. A hollow circle is a slot whose speed is
  NaN -- in practice those positions are held or interpolated rather than observed (frozen in
  place, or drifting in perfectly even steps), most often the keeper at the far end of the pitch.
- The number in a circle is the slot ID, not a shirt number and not a person. See DATA.md.
- A green ring marks a slot that has just appeared, a dashed red ring the last position of one
  just lost, and a dashed grey line joins a lost slot to a same-team slot that appears within
  2.5 s and 6 m of it -- probably the same player under a new ID.
- The strip under the pitch is possession over the whole clip: truth on top, your file below.

The game clock
--------------
Derived from the two kickoff anchors in games.py, counting 25 frames a second from each kickoff
and reading stoppage time as 45+ and 90+. Football's clock does not stop, so this is the match
clock to within a frame or two of the kickoff anchor; it is not read off the burned-in clock.

The output contains match data
------------------------------
The page embeds tracking positions, so it is written under data/ by default, which .gitignore
keeps out of git. Do not commit it or share it: the data terms cover derived data too.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import webbrowser
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import (GAME, GAMES, DATA_DIR, anchor_video_frame,  # noqa: E402
                   half_video_blocks, require, source_frames)

# Params
TEMPLATE = os.path.join(HERE, "replay_template.html")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
OUT_DIR = os.path.join(DATA_DIR, "viz")
DEFAULT_DURATION = 40.0
DEFAULT_STEP = 3                 # keep every 3rd frame; the page interpolates between them
MAX_DURATION = 15 * 60           # ~1.5 MB of page at the default step; longer gets sluggish

FPS = GAMES[GAME]["fps"]


def parse_time(text):
    """Video time as seconds, from '7:33.4', '1:14:10', '453.4' or 'f11335' (a video frame)."""
    text = text.strip()
    if text.startswith("f"):
        return int(text[1:]) / float(FPS)
    if not re.fullmatch(r"\d+(:\d{1,2})*(\.\d+)?", text):
        raise argparse.ArgumentTypeError(
            "can't read {!r} as a time; use m:ss.s, h:mm:ss, seconds, or f<frame>".format(text))
    parts = text.split(":")
    if any(float(part) >= 60 for part in parts[1:]):
        raise argparse.ArgumentTypeError("minutes and seconds after a ':' must be under 60: {!r}"
                                         .format(text))
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def read_teams(meta_path):
    """({slot: 'H' or 'A'}, {'H': name, 'A': name}) from the ATD metadata XML."""
    root = ET.parse(meta_path).getroot()
    score = root.find(".//Score")
    side_of_team = {score.get("idLocalTeam"): "H", score.get("idVisitingTeam"): "A"}
    names = {side_of_team[team.get("id")]: team.findtext("Name")
             for team in root.iter("Team") if team.get("id") in side_of_team}
    slots = {int(player.get("id")): side_of_team[player.get("teamId")]
             for player in root.iter("Player")}
    return slots, names


def read_labels(path, first, last):
    """{video_frame: label} for frames first..last of a frame,label CSV."""
    labels = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if first <= frame <= last:
                labels[frame] = row["label"].strip()
    return labels


def read_tracking(raw_path, first, last, step):
    """[[video_frame, [[slot, x, y, estimated], ...], ball_or_None], ...] for every step-th frame.

    Positions are kept in thousandths of the pitch (0..1000) to keep the page small. `estimated`
    is 1 where the slot has a position but no speed.
    """
    frames = []
    with open(raw_path) as handle:
        for line in handle:
            frame_id, players, ball = line.rstrip("\n").split(":")
            video_frame = int(frame_id) - 1          # the ATD counts from 1, the video from 0
            if video_frame < first:
                continue
            if video_frame > last:
                break
            if (video_frame - first) % step:
                continue
            slots = []
            for slot, group in enumerate(players.split(";"), start=1):
                x, y, speed = group.split(",")
                if x == "NaN":
                    continue
                slots.append([slot, round(float(x) * 1000), round(float(y) * 1000),
                              1 if speed.strip() == "NaN" else 0])
            bx, by = ball.split(",")[:2]
            frames.append([video_frame, slots,
                           None if bx == "NaN" else [round(float(bx) * 1000),
                                                     round(float(by) * 1000)]])
    return frames


def clock_spec():
    """What the page needs to show a game clock: kickoff and last frame of each half."""
    game = GAMES[GAME]
    ends = {half: last for half, _, last in half_video_blocks(GAME)}
    return {"fps": FPS,
            "kickoff": [anchor_video_frame(game["kickoff"]), anchor_video_frame(game["kickoff_2nd"])],
            "end": [ends[1], ends[2]]}


def build(start_s, duration_s, pred_path=None, gt_path=PUBLIC_GT, step=DEFAULT_STEP):
    """The page's data payload, as a dict."""
    n_frames = source_frames()
    first = max(0, int(round(start_s * FPS)))
    last = min(n_frames - 1, first + int(round(duration_s * FPS)) - 1)
    if first >= n_frames:
        raise SystemExit("start {:.1f} s is past the end of the video ({:.1f} s)"
                         .format(start_s, n_frames / float(FPS)))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    slots, names = read_teams(meta_path)
    frames = read_tracking(raw_path, first, last, step)

    truth = read_labels(gt_path, first, last) if gt_path else {}
    pred = read_labels(pred_path, first, last) if pred_path else None
    if pred is not None and not pred:
        print("warning: {} has no rows in frames {}..{} -- check it with "
              "evaluation/validate_submission.py".format(pred_path, first, last))
    for frame in frames:
        frame.append(truth.get(frame[0], "X"))
        if pred is not None:
            frame.append(pred.get(frame[0], "X"))

    return {"teams": slots, "names": names, "step": step, "first": first, "last": last,
            "clock": clock_spec(), "frames": frames,
            "gt_name": os.path.basename(gt_path) if gt_path else None,
            "pred_name": os.path.basename(pred_path) if pred_path else None}


def build_external(match, start_s, duration_s, pred_path=None, step=DEFAULT_STEP):
    """The same payload for an outside match from external/build.py. Frames are npz rows; the truth
    is that provider's own label, not the challenge rule (see external/README.md)."""
    import numpy as np
    path = os.path.join(DATA_DIR, "external", match + ".npz")
    if not os.path.exists(path):
        raise SystemExit("no {} -- build it with external/build.py".format(path))
    z = np.load(path)
    fps, (length, width), halves = int(z["fps"]), z["pitch"], z["halves"].tolist()
    n = len(z["labels"])
    first = max(0, int(round(start_s * fps)))
    last = min(n - 1, first + int(round(duration_s * fps)) - 1)
    if first >= n:
        raise SystemExit("start {:.1f} s is past the end of {} ({:.1f} s)".format(start_s, match, n / fps))

    rows = range(first, last + 1, step)
    players, ball, labels = z["players"][rows], z["ball"][rows], z["labels"][rows]
    pred = read_labels(pred_path, first, last) if pred_path else None
    frames = []
    for i, frame in enumerate(rows):
        slots = [[k + 1, round(float(x / length * 1000)), round(float(y / width * 1000)), 0]
                 for k, (x, y, _) in enumerate(players[i]) if np.isfinite(x)]
        b = ball[i]
        frames.append([frame, slots, [round(float(b[0] / length * 1000)), round(float(b[1] / width * 1000))]
                       if np.isfinite(b[0]) else None, str(labels[i])])
        if pred is not None:
            frames[-1].append(pred.get(frame, "X"))

    return {"teams": {k + 1: "HA"[t] for k, t in enumerate(z["team"].tolist())},
            "names": {"H": "Home", "A": "Away"}, "match": match, "step": step, "first": first,
            "last": last, "frames": frames, "gt_name": match + " labels",
            "clock": {"fps": fps, "kickoff": [h[0] for h in halves[:2]], "end": [h[1] for h in halves[:2]]},
            "pred_name": os.path.basename(pred_path) if pred_path else None}


def default_out(start_s, duration_s, pred_path, match=None):
    hours, rest = divmod(start_s, 3600)
    stamp = "{:d}-{:02d}-{:04.1f}".format(int(hours), int(rest // 60), rest % 60)
    name = "replay_{}{}_{:d}s".format(match + "_" if match else "", stamp, int(round(duration_s)))
    if pred_path:
        name += "_" + os.path.splitext(os.path.basename(pred_path))[0]
    return os.path.join(OUT_DIR, name + ".html")


def write_page(payload, out_path):
    with open(TEMPLATE) as handle:
        template = handle.read()
    # "</" inside a <script> would end it early; JSON allows the escaped form.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write(template.replace("__REPLAY_DATA__", data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=parse_time, required=True,
                        help="tactical video time to start at: 7:33.4, 1:14:10, 453.4 or f11335")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="seconds of video to include (default %(default)s)")
    parser.add_argument("--match", help="an outside match in data/external/, e.g. idsse_J03WMX or "
                        "metrica_1, or 'all' for BAR-ATH and every outside match with a switcher "
                        "between them; times and frames are then that match's own (needs numpy)")
    parser.add_argument("--pred", help="a frame,label submission CSV to show alongside the truth")
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground truth CSV (default: the public 5 minutes); 'none' to hide")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help="keep every n-th frame; 1 is exact but makes a bigger page")
    parser.add_argument("--out", help="where to write the page (default: data/viz/...)")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    args = parser.parse_args()

    if not 0 < args.duration <= MAX_DURATION:
        parser.error("--duration must be between 0 and {} seconds".format(MAX_DURATION))
    if args.step < 1:
        parser.error("--step must be at least 1")
    gt = None if args.gt.lower() == "none" else args.gt

    matches = [args.match]
    if args.match == "all":  # BAR-ATH plus every outside match, each page linking to the others
        if args.pred or args.out:
            parser.error("--match all can't take --pred or --out: frames and paths differ per game")
        matches = [None] + sorted(os.path.basename(f)[:-4] for f in
                                  glob.glob(os.path.join(DATA_DIR, "external", "*.npz")))
    outs = [args.out or default_out(args.start, args.duration, args.pred, m) for m in matches]
    games = [[m or GAME, os.path.basename(o)] for m, o in zip(matches, outs)]

    for match, out in zip(matches, outs):
        if match:
            payload = build_external(match, args.start, args.duration, args.pred, args.step)
        else:
            payload = build(args.start, args.duration, args.pred, gt, args.step)
        if len(games) > 1:
            payload["games"] = games
        write_page(payload, out)
        print("wrote {}  (frames {}..{}, {} samples, every {} frame{})".format(
            out, payload["first"], payload["last"], len(payload["frames"]), args.step,
            "" if args.step == 1 else "s"))
    if args.open:
        webbrowser.open("file://" + os.path.abspath(outs[0]))
