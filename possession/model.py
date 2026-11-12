"""Possession per frame from the players alone, never the ball. The simplest version:

- who: the team with the ball stretches the pitch and both teams move toward the goal it attacks.
  A small logistic model reads four shape and movement signals (TEAM_SIGNALS), each over the 2 s
  after and the 2 s before the frame, and leans home above even odds.
- dead: a stoppage changes how the whole team moves, not just how fast. A small logistic model
  reads five players-only signals (DEAD_SIGNALS) and calls the frame dead above even odds. Its weights
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
DECODE = "viterbi"  # "viterbi": the best H/A/D path below; "absorb": cut each model at even odds, absorb short runs
SWITCH_TEAM = 5.0  # what a change of team must outweigh, in log-odds summed over frames
TURN_CLUE = 0.0    # how much one TURN_SD of turn_impulse takes off SWITCH_TEAM; 0 leaves the clue out
TURN_SPAN = 0.6    # s either side for the acceleration
TURN_LAG = 0.6     # s: the players' answer to a turnover peaks this long after it
TURN_SD = 0.5      # m/s^2: the impulse's sd on the outside matches
TURN_FLOOR = 0.5   # a change of team never costs less than this
SWITCH_DEAD = 20.0  # the same for the ball going dead or coming alive
MIN_TEAM = 2.0     # s, "absorb" only: shorter H/A runs are absorbed into a neighbour
MIN_DEAD = 3.0     # s, "absorb" only: shorter D runs are absorbed into a neighbour
DEAD = "logit"     # "logit": the fitted model below; "median": median speed under WALKING
MAX_SPEED = 12.0   # m/s; faster is a tracked identity jumping, not a player (as hmm/clues.py)
MAX_ACCEL = 8.0    # m/s^2; a sharper change of speed is tracking noise, not a player
ACCEL_LAG = 5      # frames either side for a player's change of speed (0.4 s in all)

# Players-only stoppage signals, each with its own football reason:
DEAD_SIGNALS = (
    "median speed, +-0.5 s",          # the typical player walks at a stoppage
    "speed change, 2 s after - before",  # everyone slows into a stoppage, speeds up out of it
    "80th percentile |acceleration|, +-3 s",  # in play players keep changing pace; at a stoppage they don't
    "players on or over a boundary line, +-1 s",  # someone is taking a throw-in or a corner, or fetching the ball
    "both teams' width, summed, +-1 s",  # in play both sides spread to use and cover the pitch; at a stoppage they close up
)
ON_LINE = 0.5      # m inside a touchline or goal line still counts as on it
# Logistic weights on the standardised signals: intercept, then one per signal, with each signal's
# mean and sd. Fitted on all 30 outside matches, each provider weighted equally:
#   uv run python possession/model.py --fit
DEAD_FIT = {"w": [-1.312, -1.146, -0.482, -1.662, 0.775, -0.783],
            "mu": [1.744, -0.002, 0.932, 0.301, 23.72], "sd": [0.995, 0.614, 0.374, 0.61, 5.025]}

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
TEAM_FIT = {"w": [0.174, 2.175, -0.49, 0.242, 0.016, 1.197, -0.567, -0.282, -0.197],
            "mu": [-0.028, -0.028, 0.149, 0.154, 0.014, 0.017, 0.004, 0.004],
            "sd": [4.046, 4.106, 2.647, 2.679, 1.792, 1.765, 0.468, 0.465]}


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


def dead_signals(players, team):
    """(frames, 5) array of DEAD_SIGNALS, NaN where nobody is tracked."""
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
        x, y = players[:, :, 0].astype(float), players[:, :, 1].astype(float)
        edge = np.minimum(np.minimum(x, PITCH[0] - x), np.minimum(y, PITCH[1] - y))  # m inside the nearest line
        on_line = np.where(np.isfinite(x).any(1), (edge < ON_LINE).sum(1), np.nan)
        width = np.nanstd(y[:, team == 0], 1) + np.nanstd(y[:, team == 1], 1)
    return np.column_stack([smooth(q50, 0.5), window_mean(q50, 2, True) - window_mean(q50, 2, False),
                            smooth(a80, 3), smooth(on_line, 1), smooth(width, 1)])


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
    """Every fitted setting, from [(provider, players, labels, halves, team, pitch)]: what
    external/heldout.py refits without the provider it scores. Dead against live on every labelled
    frame; home against away on the frames either team has the ball."""
    global HALVES, PITCH
    dead, who, provs = ([], []), ([], []), []
    for prov, players, labels, halves, team, pitch in matches:
        HALVES, PITCH = halves, features.PITCH = halves, tuple(pitch)
        provs.append(prov)
        g = np.isin(labels, ["H", "A", "D"])
        dead[0].append(dead_signals(players, team)[g]); dead[1].append(labels[g] == "D")
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


def turn_impulse(F):
    """Per frame, how hard everybody is accelerating toward the goal home attacks, TURN_LAG
    seconds later, in units of TURN_SD: positive where the players are about to answer home
    winning the ball. Both teams turn the same way after a turnover, and that shows in their
    acceleration about 0.7 s later, well before their shape changes."""
    push = features.team_signals(F["players"], F["team"], HALVES)[:, 2]
    k, lag = int(TURN_SPAN * FPS), int(TURN_LAG * FPS)
    out = np.zeros(len(push))
    for a, b in HALVES:
        v = push[a:b + 1]
        acc = np.full(len(v), np.nan)
        acc[k:-k] = (v[2 * k:] - v[:-2 * k]) * FPS / (2 * k)
        out[a:b + 1 - lag] = np.nan_to_num(acc[lag:]) / TURN_SD
    return np.nan_to_num(smooth(out, 0.4))


def viterbi(p_home, p_dead, switch_team, switch_dead, turn=None, turn_clue=0.0):
    """The H/A/D path (as 0/1/2) with the highest total log-probability, where each frame scores
    log P(its label) from the two models and each change of label costs switch_team (H <-> A) or
    switch_dead (to or from D). One frame's evidence can't flip the label; a run of it can, and
    the change lands where the evidence turns rather than a fixed delay after."""
    with np.errstate(divide="ignore"):
        E = np.column_stack([np.log(1 - p_dead) + np.log(p_home), np.log(1 - p_dead) + np.log(1 - p_home),
                             np.log(p_dead)]).clip(-50)
    cost = np.array([[0.0, switch_team, switch_dead], [switch_team, 0, switch_dead], [switch_dead, switch_dead, 0]])
    back = np.zeros((len(E), 3), np.int8)
    score = E[0]
    if turn is not None and turn_clue:  # a change of team is cheaper where the players turn that way
        to_home = np.maximum(switch_team - turn_clue * np.maximum(turn, 0), TURN_FLOOR)
        to_away = np.maximum(switch_team - turn_clue * np.maximum(-turn, 0), TURN_FLOOR)
    for i in range(1, len(E)):
        if turn is not None and turn_clue:
            cost[1, 0], cost[0, 1] = to_home[i], to_away[i]
        via = score[:, None] - cost
        back[i] = via.argmax(0)
        score = via.max(0) + E[i]
    path = np.empty(len(E), np.int8)
    path[-1] = score.argmax()
    for i in range(len(E) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def predict(F, params=None):
    return explain(F, params)["pred"]


def explain(F, params=None):
    """predict, with its working for the viewer and for combining with another model: the labels,
    the frame-by-frame guess before clean-up, P(home | alive) and P(dead) per frame, and each
    model's signals and log-odds."""
    p = dict(keepers=KEEPERS, team_window=TEAM_WINDOW, dead_window=DEAD_WINDOW, walking=WALKING,
             min_team=MIN_TEAM, min_dead=MIN_DEAD, dead=DEAD, dead_fit=DEAD_FIT,
             team=TEAM, team_fit=TEAM_FIT, decode=DECODE, switch_team=SWITCH_TEAM,
             switch_dead=SWITCH_DEAD, turn_clue=TURN_CLUE) | (params or {})
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
        dead_x = dead_signals(F["players"], F["team"])
        dead_lo = logit(dead_x, p["dead_fit"])
        dead_p = 1 / (1 + np.exp(-dead_lo))
    else:
        dead_p = np.where(pace >= p["walking"], 0.0, 1.0)  # NaN pace counts as dead
    if "dead" in F:  # graded frames before a kickoff
        dead_p = np.where(F["dead"], 1.0, dead_p)
    alive = dead_p < 0.5
    raw = np.where(alive, pred, "D")
    turn = turn_impulse(F) if p["turn_clue"] and p["decode"] == "viterbi" else None
    for a, b in HALVES:
        if p["decode"] == "viterbi":
            pred[a:b + 1] = np.array(list("HAD"))[viterbi(home_p[a:b + 1], dead_p[a:b + 1], p["switch_team"], p["switch_dead"],
                                                          None if turn is None else turn[a:b + 1], p["turn_clue"])]
        else:
            # Blips: absorb short runs in alive/dead and in the team lean separately, so a dead
            # blip can't move a turnover: the team changes where the lean does.
            alive[a:b + 1] = absorb_short(alive[a:b + 1], {True: p["min_team"] * FPS, False: p["min_dead"] * FPS})
            pred[a:b + 1] = absorb_short(pred[a:b + 1], {"H": p["min_team"] * FPS, "A": p["min_team"] * FPS})
            pred[a:b + 1][~alive[a:b + 1]] = "D"
    pred[~graded_mask(len(pred))] = raw[~graded_mask(len(pred))] = "X"
    logit_team, logit_dead = p["team"] == "logit", p["dead"] == "logit"
    return dict(pred=pred, raw=raw, p_home=home_p, p_dead=dead_p,
                dead=dict(signals=DEAD_SIGNALS, fit=p["dead_fit"], x=dead_x) if logit_dead else None,
                team=dict(signals=[f"{s}, {when}" for s in TEAM_SIGNALS for when in ("next 2 s", "last 2 s")],
                          fit=p["team_fit"], x=team_x) if logit_team else None,
                lanes=[("team model, log-odds of home" if logit_team else "spread, home minus away, metres", "team", lean),
                       ("median player speed minus walking, m/s", "dead", pace - p["walking"])]
                + ([("stoppage model, log-odds of dead", "dead", dead_lo)] if logit_dead else []))


if __name__ == "__main__" and sys.argv[1:] == ["--fit"]:  # refit DEAD_FIT and TEAM_FIT on every outside match
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "external", "*.npz")))
    ms = []
    for path in paths:
        d = np.load(path)
        ms.append((os.path.basename(path).split("_")[0], d["players"], d["labels"], [tuple(h) for h in d["halves"]], d["team"], d["pitch"]))
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
    v = lambda ph, pd, ct=3, cd=3: "".join("HAD"[k] for k in viterbi(np.array(ph), np.array(pd), ct, cd))  # noqa: E731
    lo = [0.1] * 9
    assert v([.9, .9, .9, .9, .1, .9, .9, .9, .9], lo) == "HHHHHHHHH"   # one away frame can't pay for two switches
    assert v([.9, .9, .9, .1, .1, .1, .1, .1, .1], lo) == "HHHAAAAAA"   # a turnover lands where the evidence turns
    assert v([.9] * 9, [.1, .1, .1, .9, .1, .1, .1, .1, .1]) == "HHHHHHHHH"   # a dead blip inside a possession
    assert v([.9] * 4 + [.1] * 5, [.1, .1, .99, .99, .99, .99, .99, .1, .1]) == "HHDDDDDAA"  # a stoppage splits it
    assert v([.5] * 9, [.1] * 4 + [1.0] * 5) == "HHHHDDDDD" or v([.5] * 9, [.1] * 4 + [1.0] * 5) == "AAAADDDDD"  # forced dead
    assert v([.9, .9, .5, .5, .5, .5, .9, .9, .9], lo) == "HHHHHHHHH"   # no team evidence: stays with who had it
    print("ok")
