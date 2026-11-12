"""Possession per frame from the players alone, never the ball. The simplest version:

- who: the team with the ball stretches the pitch and both teams move toward the goal it attacks.
  A small logistic model reads four shape and movement signals (TEAM_SIGNALS), each over the 2 s
  after and the 2 s before the frame, and leans home above even odds.
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

TEAM = "logit"     # "logit": the fitted model below; "spread": the more spread-out team has it
TEAM_WINDOW = 2.0  # s after and s before the frame for each team signal (either side for "spread")
DEAD_WINDOW = 0.5  # s either side for the walking test
WALKING = 1.2      # m/s median player speed below which play is dead
KEEPERS = 0        # 1 counts each team's deepest player (the keeper) in its spread
MIN_TEAM = 2.0     # s: shorter H/A runs are absorbed into a neighbour
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
# Logistic weights on the standardised signals: intercept, then one per signal, with each signal's
# mean and sd. Fitted on all 30 outside matches, each provider weighted equally:
#   uv run python possession/model.py --fit
DEAD_FIT = {"w": [-1.381, -1.462, -0.481, -1.578],
            "mu": [1.744, -0.002, 0.932], "sd": [0.995, 0.614, 0.374]}

# Players-only who-has-it signals (features.team_signals), home minus away. Each is read twice:
# over the TEAM_WINDOW after the frame and over the one before, because shape answers a turnover
# late: what the teams do next says more about who has the ball now than what they just did.
TEAM_SIGNALS = (
    "width, home - away, m",           # the side with the ball stretches the pitch
    "length, home - away, m",          # and is strung out along it
    "push toward away's goal, m/s",    # both teams move the way the ball is going
    "mean speed, home - away, m/s",    # the side without the ball chases
)
# As DEAD_FIT: intercept, then after and before for each signal in turn.
TEAM_FIT = {"w": [0.164, 2.168, -0.366, 0.597, -0.496, 0.912, -0.456, -0.321, -0.184],
            "mu": [-0.027, -0.026, 0.297, 0.296, 0.001, 0.005, 0.006, 0.006],
            "sd": [4.028, 4.086, 3.24, 3.256, 1.781, 1.755, 0.47, 0.467]}


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


def dead_prob(X, fit=DEAD_FIT):
    """P(dead) per frame."""
    return 1 / (1 + np.exp(-logit(X, fit)))


def team_signals(F):
    """(frames, 8): each of TEAM_SIGNALS over the TEAM_WINDOW after the frame, then before it."""
    raw = features.team_signals(F["players"], F["team"], HALVES)
    return np.column_stack([window_mean(x, TEAM_WINDOW, ahead) for x in raw.T for ahead in (True, False)])


def logit(X, fit):
    """Log-odds per frame from a fitted logistic. A missing signal (a window with nobody tracked,
    or past the edge of a half) adds nothing either way; with none at all it is the intercept."""
    return fit["w"][0] + np.nan_to_num((X - fit["mu"]) / fit["sd"]) @ np.array(fit["w"][1:])


def fit_logistic(Xs, ys, provs, l2=1e-3):
    """Logistic weights on standardised signals from per-match (X, y), each match's frames
    weighted so every provider counts equally. Newton's method on the log-likelihood."""
    per = {p: provs.count(p) for p in provs}
    ws = []
    for i, (X, y, prov) in enumerate(zip(Xs, ys, provs)):
        ok = np.isfinite(X).all(1)
        Xs[i], ys[i] = X[ok], y[ok]
        ws.append(np.full(ok.sum(), 1 / (per[prov] * ok.sum())))
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
    """Every fitted setting, from [(provider, players, labels, halves, team)]: what
    external/heldout.py refits without the provider it scores. Dead against live on every labelled
    frame; home against away on the frames either team has the ball."""
    global HALVES
    dead, who, provs = ([], []), ([], []), []
    for prov, players, labels, halves, team in matches:
        HALVES = halves
        provs.append(prov)
        g = np.isin(labels, ["H", "A", "D"])
        dead[0].append(dead_signals(players)[g]); dead[1].append(labels[g] == "D")
        g = np.isin(labels, ["H", "A"])
        who[0].append(team_signals(dict(players=players, team=team))[g]); who[1].append(labels[g] == "H")
    return {"dead_fit": fit_logistic(*dead, provs), "team_fit": fit_logistic(*who, provs)}


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
    """predict, with its working for the viewer and for combining with another model: the labels,
    the frame-by-frame guess before clean-up, P(home | alive) and P(dead) per frame, and each
    model's signals and log-odds."""
    p = dict(keepers=KEEPERS, team_window=TEAM_WINDOW, dead_window=DEAD_WINDOW, walking=WALKING,
             min_team=MIN_TEAM, min_dead=MIN_DEAD, dead=DEAD, dead_fit=DEAD_FIT,
             team=TEAM, team_fit=TEAM_FIT) | (params or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nobody tracked
        pace = smooth(np.nanmedian(F["players"][:, :, 2], 1), p["dead_window"])
    if p["team"] == "logit":
        team_x = team_signals(F)
        lean = np.where(np.isfinite(team_x).any(1), logit(team_x, p["team_fit"]), np.nan)  # log-odds of home
        home_p = 1 / (1 + np.exp(-np.nan_to_num(lean)))
    else:
        sh, sa = features.spread(F["players"], F["team"], HALVES, keepers=bool(p["keepers"]))
        lean = smooth(sh - sa, p["team_window"])  # metres
        home_p = np.where(np.nan_to_num(lean) >= 0, 1.0, 0.0)
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
    else:
        dead_p = np.where(pace >= p["walking"], 0.0, 1.0)  # NaN pace counts as dead
    alive = dead_p < 0.5
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
    logit_team, logit_dead = p["team"] == "logit", p["dead"] == "logit"
    return dict(pred=pred, raw=raw, p_home=home_p, p_dead=dead_p,
                dead=dict(signals=DEAD_SIGNALS, fit=p["dead_fit"], x=dead_x) if logit_dead else None,
                team=dict(signals=[f"{s}, {when}" for s in TEAM_SIGNALS for when in ("next 2 s", "last 2 s")],
                          fit=p["team_fit"], x=team_x) if logit_team else None,
                lanes=[("team model, log-odds of home" if logit_team else "spread, home minus away, metres", "team", lean),
                       ("median player speed minus walking, m/s", "dead", pace - p["walking"])]
                + ([("stoppage model, log-odds of dead", "dead", np.log(dead_p / (1 - dead_p)))] if logit_dead else []))


if __name__ == "__main__" and sys.argv[1:] == ["--fit"]:  # refit DEAD_FIT and TEAM_FIT on every outside match
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "external", "*.npz")))
    ms = []
    for path in paths:
        d = np.load(path)
        ms.append((os.path.basename(path).split("_")[0], d["players"], d["labels"], [tuple(h) for h in d["halves"]], d["team"]))
    for name, value in fit(ms).items():
        print(name.upper(), "=", value)
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
