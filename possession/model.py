"""Possession per frame, the simplest version: the team whose nearest player is closer to the ball
has it. Where the ball isn't seen, D. No smoothing.

This is the starting point to build on one idea at a time, each one kept only if it raises the
score on the outside matches (external/score.py).
"""

import numpy as np

from load import FPS, PITCH  # noqa: F401  (external/score.py repoints PITCH and HALVES per match)

HALVES = None  # [(first, last)] graded frames per half, set by the caller


def graded_mask(n):
    mask = np.zeros(n, bool)
    for a, b in HALVES:
        mask[a:b + 1] = True
    return mask


def predict(F, params=None):
    return explain(F, params)["pred"]


def explain(F, params=None):
    """predict, with its working for the viewer: the one clue's lean per frame, in metres --
    how much closer home's nearest player is than away's (positive leans home)."""
    gap = F["da"] - F["dh"]
    pred = np.where(np.isfinite(gap), np.where(gap >= 0, "H", "A"), "D")
    if "dead" in F:  # graded frames before a kickoff
        pred[F["dead"]] = "D"
    pred[~graded_mask(len(pred))] = "X"
    return dict(pred=pred, raw=pred, lanes=[("nearest player, metres", "team", gap)])
