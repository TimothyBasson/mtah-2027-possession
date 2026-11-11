"""Possession per frame from the players alone, never the ball. The simplest version:

- who: the more spread-out team has the ball (attackers stretch, defenders compact)
- dead: a stoppage changes how the whole team moves, not just how fast. A small logistic model
  reads four speed signals (DEAD_SIGNALS) and calls the frame dead above even odds. Its weights
  are fitted on the outside matches; external/heldout.py refits them without the provider it scores.

Both are smoothed over a window centred on the frame, which is allowed because the task is
offline. Windows stop at the edges of each half.

This is the starting point to build on one idea at a time, each one kept only if it raises the
score on the outside matches (external/score.py).
"""

import itertools
import sys
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
DEAD = "logit"     # "logit": the fitted model below; "median": median speed under WALKING
MAX_SPEED = 12.0   # m/s; faster is a tracked identity jumping, not a player (as hmm/clues.py)

# Players-only stoppage signals, each with its own football reason:
DEAD_SIGNALS = (
    "median speed, +-0.5 s",          # the typical player walks at a stoppage
    "80th percentile speed, +-0.5 s",  # in slow build-up the fastest few still run
    "median speed, +-10 s",           # stoppages last; a slow patch in play doesn't
    "speed change, 2 s after - before",  # everyone slows into a stoppage, speeds up out of it
)
# Logistic weights on the standardised signals: intercept, then one per signal, with each signal's
# mean and sd. Fitted on all 30 outside matches, each provider weighted equally:
#   uv run python possession/model.py --fit
DEAD_FIT = {"w": [-1.255, 0.271, -1.785, -1.004, -0.486],
            "mu": [1.744, 2.567, 1.75, -0.002], "sd": [0.995, 1.269, 0.666, 0.614]}


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


def window_mean(x, seconds, ahead):
    """Mean over the `seconds` just after (ahead) or just before each frame, within each half."""
    n = int(seconds * FPS)
    out = np.full(len(x), np.nan)
    for a, b in HALVES:
        v = x[a:b + 1]
        c, k = np.r_[0, np.cumsum(np.nan_to_num(v))], np.r_[0, np.cumsum(np.isfinite(v))]
        i = np.arange(len(v))
        lo, hi = (i + 1, np.minimum(len(v), i + 1 + n)) if ahead else (np.maximum(0, i - n), i)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b + 1] = (c[hi] - c[lo]) / (k[hi] - k[lo])
    return out


def dead_signals(players):
    """(frames, 4) array of DEAD_SIGNALS, NaN where nobody is tracked."""
    s = players[:, :, 2].astype(float)
    s[s > MAX_SPEED] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nobody tracked
        q50, q80 = np.nanmedian(s, 1), np.nanpercentile(s, 80, 1)
    return np.column_stack([smooth(q50, 0.5), smooth(q80, 0.5), smooth(q50, 10),
                            window_mean(q50, 2, True) - window_mean(q50, 2, False)])


def dead_prob(X, fit=DEAD_FIT):
    z = fit["w"][0] + ((X - fit["mu"]) / fit["sd"]) @ np.array(fit["w"][1:])
    return 1 / (1 + np.exp(-z))


def fit_dead(matches, l2=1e-3):
    """DEAD_FIT from [(players, labels, halves)], each match's frames weighted so every provider
    counts equally: [(provider, players, labels, halves)]. Newton's method on the log-likelihood."""
    global HALVES
    Xs, ys, ws = [], [], []
    per = {}
    for prov, *_ in matches:
        per[prov] = per.get(prov, 0) + 1
    for prov, players, labels, halves in matches:
        HALVES = halves
        g = np.isin(labels, ["H", "A", "D"])
        X = dead_signals(players)[g]
        ok = np.isfinite(X).all(1)
        Xs.append(X[ok]); ys.append((labels[g] == "D")[ok]); ws.append(np.full(ok.sum(), 1 / (per[prov] * ok.sum())))
    X, y, wt = np.vstack(Xs), np.concatenate(ys).astype(float), np.concatenate(ws)
    wt /= wt.sum()
    mu, sd = X.T @ wt, np.sqrt(((X - X.T @ wt) ** 2).T @ wt)
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (wt * (p - y)) + l2 * w
        hess = (Z * (wt * p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return {k: [round(float(x), 3) for x in v] for k, v in (("w", w), ("mu", mu), ("sd", sd))}


def fit(matches):
    """Every fitted setting, from [(provider, players, labels, halves)]: what external/heldout.py
    refits without the provider it scores."""
    return {"dead_fit": fit_dead(matches)}


def absorb_short(labels, min_len):
    """Relabel runs shorter than min_len[label] frames as a neighbour, shortest first, so one
    brief flip doesn't split a possession in two. Takes the longer neighbour; the model only
    passes two-valued labels, where both neighbours are the same."""
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
             min_team=MIN_TEAM, min_dead=MIN_DEAD, dead=DEAD, dead_fit=DEAD_FIT) | (params or {})
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
    if p["dead"] == "logit":
        dead_x = dead_signals(F["players"])
        dead_p = dead_prob(dead_x, p["dead_fit"])
        alive = dead_p < 0.5  # NaN (nobody tracked) counts as dead
    else:
        alive = pace >= p["walking"]  # NaN pace counts as dead
    if "dead" in F:  # graded frames before a kickoff
        alive[F["dead"]] = False
    raw = np.where(alive, pred, "D")
    # Blips: absorb short runs in alive/dead and in the team lean separately, so a dead blip can't
    # move a turnover: the team changes where the lean does.
    for a, b in HALVES:
        alive[a:b + 1] = absorb_short(alive[a:b + 1], {True: p["min_team"] * FPS, False: p["min_dead"] * FPS})
        pred[a:b + 1] = absorb_short(pred[a:b + 1], {"H": p["min_team"] * FPS, "A": p["min_team"] * FPS})
    pred[~alive] = "D"
    pred[~graded_mask(len(pred))] = raw[~graded_mask(len(pred))] = "X"
    return dict(pred=pred, raw=raw,
                dead=dict(signals=DEAD_SIGNALS, fit=p["dead_fit"], x=dead_x) if p["dead"] == "logit" else None,
                lanes=[("spread, home minus away, metres", "team", lean),
                                            ("median player speed minus walking, m/s", "dead", pace - p["walking"])]
                + ([("stoppage model, log-odds of dead", "dead", np.log(dead_p / (1 - dead_p)))] if p["dead"] == "logit" else []))


if __name__ == "__main__" and sys.argv[1:] == ["--fit"]:  # refit DEAD_FIT on every outside match
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "external", "*.npz")))
    ms = []
    for path in paths:
        d = np.load(path)
        ms.append((os.path.basename(path).split("_")[0], d["players"], d["labels"], [tuple(h) for h in d["halves"]]))
    print("DEAD_FIT =", fit(ms)["dead_fit"])
elif __name__ == "__main__":  # uv run python possession/model.py: check the run merging
    m = {"H": 3, "A": 3, "D": 2}
    f = lambda s: "".join(absorb_short(np.array(list(s)), m))  # noqa: E731
    assert f("HHHHDHHHH") == "HHHHHHHHH"          # a dead blip inside a possession
    assert f("HHHHAHHHH") == "HHHHHHHHH"          # a one-frame team flip
    assert f("HHHHHAADDD") == "HHHHHHHDDD"        # short run goes to the longer neighbour
    assert f("HHHDDAAAA") == "HHHDDAAAA"          # long enough runs are left alone
    assert f("AHHHH") == "HHHHH"                  # a run at the edge has one neighbour
    assert f("HA") == "AA" or f("HA") == "HH"     # everything short still ends as one run
    alive = absorb_short(np.array([1, 1, 1, 0, 1, 1, 0, 0, 0], bool), {True: 3, False: 2})
    assert alive.tolist() == [True] * 6 + [False] * 3  # two-valued, as the model calls it
    print("ok")
