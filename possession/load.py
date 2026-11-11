"""BAR-ATH's tracking in the same arrays external/build.py writes for the outside matches, so one
model code path serves both:

    players  (n, K, 3)  x, y in metres from a corner, speed m/s; NaN where missing
    ball     (n, 3)     x, y in metres, speed; NaN where missing
    team     (K,)       0 home, 1 away
    halves   [(first, last)] graded video frames of each half, inclusive

Parsing the 199k-line ATD takes a minute, so the arrays are cached under data/cache/ (gitignored).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
from build import speed  # noqa: E402  (external/build.py: the same speed as the outside matches)

FPS = GAMES[GAME]["fps"]
PITCH = tuple(GAMES[GAME]["pitch"])
CACHE = os.path.join(DATA_DIR, "cache", "possession_bar_ath_v1.npz")


def parse(raw_path, n_frames, n_slots):
    """Raw ATD lines "<frame>:<x,y,s;...>:<x,y,s>", coordinates 0-1, to metres."""
    xy = np.full((n_frames, n_slots + 1, 2), np.nan, np.float32)  # last column is the ball
    with open(raw_path) as handle:
        for line in handle:
            frame, players, ball = line.rstrip("\n").split(":")
            groups = players.split(";") + [ball]
            xy[int(frame) - 1] = [g.split(",")[:2] for g in groups]  # "NaN" parses to nan
    xy *= np.array(PITCH, np.float32)
    return xy[:, :-1], xy[:, -1]


def load():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["players"], z["ball"], z["team"], [tuple(h) for h in z["halves"]]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    pxy, bxy = parse(raw_path, source_frames(), len(team))
    players = np.concatenate([pxy, speed(pxy, FPS)[..., None]], -1)
    ball = np.concatenate([bxy, speed(bxy, FPS)[:, None]], -1)
    halves = [(a, b) for _, a, b in half_video_blocks(GAME)]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, players=players, ball=ball, team=team, halves=np.array(halves))
    return players, ball, team, halves
