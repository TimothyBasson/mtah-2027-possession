"""Possession per frame from the players alone, never the ball. The simplest version:

- who: a small logistic model of three players-only signals (TEAM_SIGNALS): the team with the ball
  is the more spread out and the wider, and the whole block moves toward the goal it attacks
- dead: a stoppage changes how the whole team moves, not just how fast. A small logistic model
  reads three player-movement signals (DEAD_SIGNALS) and calls the frame dead above even odds. Its weights
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

TEAM_WINDOW = 1.0  # s either side for the spread lean
DEAD_WINDOW = 0.5  # s either side for the walking test
WALKING = 1.2      # m/s median player speed below which play is dead
MIN_TEAM = 1.0     # s: shorter H/A runs are absorbed into a neighbour
MIN_DEAD = 3.0     # s: shorter D runs are absorbed into a neighbour
DEAD = "logit"     # "logit": the fitted model below; "median": median speed under WALKING
MAX_SPEED = 12.0   # m/s; faster is a tracked identity jumping, not a player (as hmm/clues.py)
MAX_ACCEL = 8.0    # m/s^2; a sharper change of speed is tracking noise, not a player
ACCEL_LAG = 5      # frames either side for a player's change of speed (0.4 s in all)

# Players-only stoppage signals, each with its own football reason:
DEAD_SIGNALS = (
    "median speed, +-0.5 s",          # the typical player walks at a stoppage
    "speed change, 2 s after - before",  # everyone slows into a stoppage, speeds up out of it
    "80th percentile |acceleration|, +-3 s",  # in play players keep changing pace; at a stoppage they don't
)
# Players-only team signals, each home minus away (or toward home's attack), each with its football reason:
TEAM_SIGNALS = (
    "spread, home - away, m",           # attackers stretch out, defenders stay compact
    "width, home - away, m",            # make the pitch big with the ball, small without it
    "forward velocity toward away goal, m/s",  # the team with the ball pushes up, the other drops
)
# Logistic weights on the signals divided by their sd, no intercept and no centring: nothing in
# the game favours home, so zero on every signal is even odds. Fitted like DEAD_FIT.
TEAM_FIT = {"w": [0.455, 1.381, 0.465], "sd": [4.289, 14.813, 1.679]}

# Logistic weights on the standardised signals: intercept, then one per signal, with each signal's
# mean and sd. Fitted on all 30 outside matches, each provider weighted equally:
#   uv run python possession/model.py --fit
DEAD_FIT = {"w": [-1.381, -1.462, -0.481, -1.578],
            "mu": [1.744, -0.002, 0.932], "sd": [0.995, 0.614, 0.374]}


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
    """(frames, 3) array of DEAD_SIGNALS, NaN where nobody is tracked."""
    s = players[:, :, 2].astype(float)
    s[s > MAX_SPEED] = np.nan
    acc = np.full(s.shape, np.nan)  # each player's change of speed, m/s^2, within each half
    k = ACCEL_LAG
    for a, b in HALVES:
        acc[a + k:b + 1 - k] = np.abs(s[a + 2 * k:b + 1] - s[a:b + 1 - 2 * k]) * FPS / (2 * k)
    acc[acc > MAX_ACCEL] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nobody tracked
        q50, a80 = np.nanmedian(s, 1), np.nanpercentile(acc, 80, 1)
    return np.column_stack([smooth(q50, 0.5), window_mean(q50, 2, True) - window_mean(q50, 2, False),
                            smooth(a80, 3)])


def team_signals(players, team, window):
    """(frames, 3) array of TEAM_SIGNALS, each smoothed over +-window s, NaN where a side is untracked."""
    sh, sa = features.spread(players, team, HALVES)
    wh, wa = features.width(players, team)
    v = features.forward_velocity(players, team, HALVES, ACCEL_LAG, FPS, MAX_SPEED)
    return np.column_stack([smooth(sh - sa, window), smooth(wh - wa, window), smooth(v, window)])


def team_logodds(X, fit):
    """Log-odds that home has the ball. A missing signal adds nothing; NaN only where all are."""
    z = np.nan_to_num(X / fit["sd"]) @ np.array(fit["w"])
    z[~np.isfinite(X).any(1)] = np.nan
    return z


def logistic(Xs, ys, provs, l2=1e-3, intercept=True):
    """Weights, mean and sd of a logistic regression on per-match signal arrays, frames weighted so
    every provider counts equally (each match equally within one). Newton's method."""
    per = {}
    for p in provs:
        per[p] = per.get(p, 0) + 1
    ok = [np.isfinite(X).all(1) for X in Xs]
    wt = np.concatenate([np.full(o.sum(), 1 / (per[p] * o.sum())) for o, p in zip(ok, provs)])
    X = np.vstack([X[o] for X, o in zip(Xs, ok)])
    y = np.concatenate([y[o] for y, o in zip(ys, ok)]).astype(float)
    wt /= wt.sum()
    mu = X.T @ wt if intercept else np.zeros(X.shape[1])
    sd = np.sqrt(((X - mu) ** 2).T @ wt)
    Z = (X - mu) / sd
    if intercept:
        Z = np.c_[np.ones(len(X)), Z]
    w = np.zeros(Z.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (wt * (p - y)) + l2 * w
        hess = (Z * (wt * p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    r = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    return dict(w=r(w), mu=r(mu), sd=r(sd)) if intercept else dict(w=r(w), sd=r(sd))


def dead_prob(X, fit=DEAD_FIT):
    """P(dead) per frame. A missing signal (a window with nobody tracked, or past the edge of a
    half) adds nothing either way, rather than forcing dead; with none at all it is the intercept."""
    z = fit["w"][0] + np.nan_to_num((X - fit["mu"]) / fit["sd"]) @ np.array(fit["w"][1:])
    return 1 / (1 + np.exp(-z))


def fit_dead(matches):
    """DEAD_FIT from [(provider, players, team, labels, halves, pitch)]: graded frames, dead or not."""
    global HALVES
    Xs, ys = [], []
    for prov, players, team, labels, halves, pitch in matches:
        HALVES = halves
        g = np.isin(labels, ["H", "A", "D"])
        Xs.append(dead_signals(players)[g]); ys.append(labels[g] == "D")
    return logistic(Xs, ys, [m[0] for m in matches])


def fit_team(matches):
    """TEAM_FIT from the same matches: frames in play, home or away."""
    global HALVES
    Xs, ys = [], []
    for prov, players, team, labels, halves, pitch in matches:
        HALVES, features.PITCH = halves, pitch
        g = np.isin(labels, ["H", "A"])
        Xs.append(team_signals(players, team, TEAM_WINDOW)[g]); ys.append(labels[g] == "H")
    return logistic(Xs, ys, [m[0] for m in matches], intercept=False)


def fit(matches):
    """Every fitted setting, from [(provider, players, team, labels, halves, pitch)]: what
    external/heldout.py refits without the provider it scores."""
    return {"dead_fit": fit_dead(matches), "team_fit": fit_team(matches)}


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
    p = dict(team_window=TEAM_WINDOW, dead_window=DEAD_WINDOW, walking=WALKING,
             min_team=MIN_TEAM, min_dead=MIN_DEAD, dead=DEAD, dead_fit=DEAD_FIT, team_fit=TEAM_FIT) | (params or {})
    team_x = team_signals(F["players"], F["team"], p["team_window"])
    lean = team_logodds(team_x, p["team_fit"])
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
        alive = dead_p < 0.5
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
                team=dict(signals=TEAM_SIGNALS, fit=p["team_fit"], x=team_x),
                lanes=[("team model, log-odds of home", "team", lean),
                                            ("median player speed minus walking, m/s", "dead", pace - p["walking"])]
                + ([("stoppage model, log-odds of dead", "dead", np.log(dead_p / (1 - dead_p)))] if p["dead"] == "logit" else []))


if __name__ == "__main__" and sys.argv[1:] == ["--fit"]:  # refit DEAD_FIT and TEAM_FIT on every outside match
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "external", "*.npz")))
    ms = []
    for path in paths:
        d = np.load(path)
        ms.append((os.path.basename(path).split("_")[0], d["players"], d["team"], d["labels"],
                   [tuple(h) for h in d["halves"]], tuple(d["pitch"])))
    f = fit(ms)
    print("DEAD_FIT =", f["dead_fit"])
    print("TEAM_FIT =", f["team_fit"])
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
