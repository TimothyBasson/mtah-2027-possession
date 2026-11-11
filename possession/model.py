"""Possession per frame from the players alone, never the ball. The simplest version:

- who: the more spread-out team has the ball (attackers stretch, defenders compact)
- dead: when the median player is walking, the ball is out of play

Both are smoothed over a window centred on the frame, which is allowed because the task is
offline. Windows stop at the edges of each half.

This is the starting point to build on one idea at a time, each one kept only if it raises the
score on the outside matches (external/score.py).
"""

import itertools
import warnings

import numpy as np

import features
from load import FPS, PITCH  # noqa: F401  (external/score.py repoints PITCH and HALVES per match)

HALVES = None  # [(first, last)] graded frames per half, set by the caller

TEAM_WINDOW = 2.0  # s either side for the spread lean
DEAD_WINDOW = 0.5  # s either side for the walking test
WALKING = 1.2      # m/s median player speed below which play is dead
KEEPERS = 0        # 1 counts each team's deepest player (the keeper) in its spread
MIN_TEAM = 2.0     # s: shorter H/A runs are absorbed into a neighbour
MIN_DEAD = 3.0     # s: shorter D runs are absorbed into a neighbour


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


def absorb_short(labels, min_len):
    """Relabel runs shorter than min_len[label] frames as a neighbour, shortest first, so one
    brief flip doesn't split a possession in two. Takes the longer neighbour."""
    runs = [[k, len(list(g))] for k, g in itertools.groupby(labels)]
    while len(runs) > 1:
        short = [i for i, (k, n) in enumerate(runs) if n < min_len.get(k, 0)]
        if not short:
            break
        i = min(short, key=lambda i: runs[i][1])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: runs[j][1])
        runs[j][1] += runs.pop(i)[1]
        if 0 < i < len(runs) and runs[i - 1][0] == runs[i][0]:  # the two sides now touch
            runs[i - 1][1] += runs.pop(i)[1]
    return np.array([k for k, n in runs for _ in range(n)], dtype=labels.dtype)


def predict(F, params=None):
    return explain(F, params)["pred"]


def explain(F, params=None):
    """predict, with its working for the viewer: how much more spread out home is than away, in
    metres (positive leans home), and the median player speed in m/s."""
    p = dict(keepers=KEEPERS, team_window=TEAM_WINDOW, dead_window=DEAD_WINDOW, walking=WALKING,
             min_team=MIN_TEAM, min_dead=MIN_DEAD) | (params or {})
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
    raw = pred.copy()
    min_len = {"H": p["min_team"] * FPS, "A": p["min_team"] * FPS, "D": p["min_dead"] * FPS}
    for a, b in HALVES:
        pred[a:b + 1] = absorb_short(pred[a:b + 1], min_len)
    if "dead" in F:  # graded frames before a kickoff
        pred[F["dead"]] = "D"
    pred[~graded_mask(len(pred))] = raw[~graded_mask(len(pred))] = "X"
    return dict(pred=pred, raw=raw, lanes=[("spread, home minus away, metres", "team", lean),
                                            ("median player speed minus walking, m/s", "dead", pace - p["walking"])])


if __name__ == "__main__":  # uv run python possession/model.py: check the run merging
    m = {"H": 3, "A": 3, "D": 2}
    f = lambda s: "".join(absorb_short(np.array(list(s)), m))  # noqa: E731
    assert f("HHHHDHHHH") == "HHHHHHHHH"          # a dead blip inside a possession
    assert f("HHHHAHHHH") == "HHHHHHHHH"          # a one-frame team flip
    assert f("HHHHHAADDD") == "HHHHHHHDDD"        # short run goes to the longer neighbour
    assert f("HHHDDAAAA") == "HHHDDAAAA"          # long enough runs are left alone
    assert f("AHHHH") == "HHHHH"                  # a run at the edge has one neighbour
    assert f("HA") == "AA" or f("HA") == "HH"     # everything short still ends as one run
    print("ok")
