"""Shared loading for the tag-based BAR-ATH check: candidate files, tags, graded frames."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FPS = 25
HALVES = [(11185, 90651), (110808, 190197)]
PUBLIC = (11185, 18684)
N = 199319
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
TEAM = {"barcelona": 0, "atletic club": 1}   # home (H), away (A)
# codes whose Team tag names the side with the ball
ATTACK = ("build up", "progression", "creating chances", "attacking transition")
PLACE = ("ball in final third", "ball in the box", "players in the box", "long ball")


def read(path):
    import pandas as pd
    lab = pd.read_csv(path)["label"].values
    assert len(lab) == N, path
    return np.array([CODE[x] for x in lab], dtype=np.int8)


def graded():
    g = np.zeros(N, bool)
    for a, b in HALVES:
        g[a:b + 1] = True
    return g


def tags():
    from starter.load_smart_tagging import load
    out = []
    for t in load():
        team = TEAM.get((t["tags"].get("Team") or "").strip().lower(), -1)
        out.append(dict(code=t["code"].strip().lower(), a=t["start_frame"], b=t["end_frame"],
                        team=team, type=(t["tags"].get("Type") or "").strip().lower()))
    return sorted(out, key=lambda t: t["a"])


def public_gt():
    return read(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
