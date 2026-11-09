"""Possession per frame from the players alone, never the ball. The simplest version:

- who: the more spread-out team has the ball (attackers stretch, defenders compact)
- dead: when the median player is walking, the ball is out of play

Both are smoothed over a window centred on the frame, which is allowed because the task is
offline. Windows stop at the edges of each half.

This is the starting point to build on one idea at a time, each one kept only if it raises the
score on the outside matches (external/score.py).
"""

import warnings

import numpy as np

import features
from load import FPS, PITCH  # noqa: F401  (external/score.py repoints PITCH and HALVES per match)

HALVES = None  # [(first, last)] graded frames per half, set by the caller

TEAM_WINDOW = 2.0  # s either side for the spread lean
DEAD_WINDOW = 0.5  # s either side for the walking test
WALKING = 1.2      # m/s median player speed below which play is dead
KEEPERS = 0        # 1 counts each team's deepest player (the keeper) in its spread


def graded_mask(n):
    mask = np.zeros(n, bool)
    for a, b in HALVES:
        mask[a:b + 1] = True
    return mask


def smooth(x, seconds):
    """Centred mean over +-seconds within each half, skipping NaNs (NaN if the window has none)."""
    n = int(seconds * FPS)
    k = np.ones(2 * n + 1)
    out = np.full(len(x), np.nan)
    for a, b in HALVES:
        v = x[a:b + 1]
        ok = np.isfinite(v)
        c = np.convolve(ok, k, "same")
        with np.errstate(invalid="ignore"):
            out[a:b + 1] = np.where(c > 0, np.convolve(np.where(ok, v, 0), k, "same") / c, np.nan)
    return out


def predict(F, params=None):
    return explain(F, params)["pred"]


def explain(F, params=None):
    """predict, with its working for the viewer: how much more spread out home is than away, in
    metres (positive leans home), and the median player speed in m/s."""
    p = dict(keepers=KEEPERS, team_window=TEAM_WINDOW, dead_window=DEAD_WINDOW, walking=WALKING) | (params or {})
    sh, sa = features.spread(F["players"], F["team"], HALVES, keepers=bool(p["keepers"]))
    lean = smooth(sh - sa, p["team_window"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nobody tracked
        pace = smooth(np.nanmedian(F["players"][:, :, 2], 1), p["dead_window"])
    pred = np.where(lean >= 0, "H", "A")
    # No lean where a side is untracked for the whole window: possession stays with the team that
    # had it (CHALLENGE.md), so carry the last lean in this half, or the first one at its start.
    for a, b in HALVES:
        ok = np.flatnonzero(np.isfinite(lean[a:b + 1]))
        if len(ok):
            last = ok[np.clip(np.searchsorted(ok, np.arange(b + 1 - a), "right") - 1, 0, None)]
            pred[a:b + 1] = pred[a:b + 1][last]
    pred[~(pace >= p["walking"])] = "D"  # NaN pace counts as dead
    if "dead" in F:  # graded frames before a kickoff
        pred[F["dead"]] = "D"
    pred[~graded_mask(len(pred))] = "X"
    return dict(pred=pred, raw=pred, lanes=[("spread, home minus away, metres", "team", lean),
                                            ("median player speed minus walking, m/s", "dead", pace - p["walking"])])
