"""Possession per frame: score each frame's evidence for H, A and D, then pick the best sequence
for each half with Viterbi, paying a fixed cost for every change of state.

Why this shape: spellF1 grades whole possessions, so a label that flickers for a second costs a
false spell. The switch cost is how strongly a change must be supported before we accept it.

Every clue gives a log-probability per state per frame (higher = more likely). They are summed,
then scaled by `weight`, because 25 neighbouring frames are not 25 independent looks at the ball.

    p = dict(DEFAULTS, **params)   # any of these can be overridden from external/score.py
"""

import numpy as np

from load import FPS, PITCH  # noqa: F401  (external/score.py repoints PITCH and HALVES per match)

HALVES = None  # [(first, last)] graded frames per half, set by the caller
STATES = np.array(["H", "A", "D"])

DEFAULTS = dict(
    gap_scale=2.0,    # m: a nearest-player gap this big is e:1 odds for that team
    out_margin=1.0,   # m past a line before a seen ball counts as out
    out_odds=8.0,     # log-odds D vs live on a frame where the ball is seen out
    speed_mid=1.3,    # m/s mean player speed where live and dead are equally likely
    speed_scale=0.3,  # m/s
    weight=0.1,       # trust in one frame's evidence
    team_cost=2.5,    # log cost of H <-> A
    dead_cost=2.0,    # log cost of live -> D and of D -> live
)


def graded_mask(n):
    mask = np.zeros(n, bool)
    for a, b in HALVES:
        mask[a:b + 1] = True
    return mask


def log_sigmoid(x):
    return -np.logaddexp(0, -x)


def clues(F, p):
    """Each clue's (n, 3) log-scores for H, A, D, before WEIGHT. Their sum is the evidence."""
    n = len(F["ball"])
    out = {}
    # Who is nearer the ball. Says nothing about D, so D gets the even-odds score.
    e = np.zeros((n, 3))
    gap = (F["da"] - F["dh"]) / p["gap_scale"]
    ok = np.isfinite(gap)
    e[ok, 0] = log_sigmoid(gap[ok])
    e[ok, 1] = log_sigmoid(-gap[ok])
    e[ok, 2] = np.log(0.5)
    out["nearest"] = e
    # A ball seen past a line is dead.
    e = np.zeros((n, 3))
    L, W = PITCH
    bx, by = F["ball"][:, 0], F["ball"][:, 1]
    with np.errstate(invalid="ignore"):
        past = np.fmax.reduce([-bx, bx - L, -by, by - W])
    e[past > p["out_margin"], 2] = p["out_odds"]
    out["ball out"] = e
    # Players jog or run in play and walk at a stoppage.
    e = np.zeros((n, 3))
    v = (F["mean_speed"] - p["speed_mid"]) / p["speed_scale"]
    ok = np.isfinite(v)
    e[ok, :2] = log_sigmoid(v[ok])[:, None]
    e[ok, 2] = log_sigmoid(-v[ok])
    out["player speed"] = e
    return out


def viterbi(e, cost):
    """Best state path through (n, 3) log-scores, with cost[i, j] paid on moving from i to j."""
    n = len(e)
    back = np.zeros((n, 3), np.int8)
    score = e[0].copy()
    trans = -cost
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(F, params=None):
    return explain(F, params)["pred"]


def explain(F, params=None):
    """predict, with its working: the frame-by-frame best guess before smoothing, and each clue's
    lean per frame as log-odds -- H over A for clues about who has the ball, D over live for clues
    about whether it is in play. A clue that says nothing on a frame leans 0 there."""
    p = dict(DEFAULTS, **(params or {}))
    cs = clues(F, p)
    e = sum(cs.values()) * p["weight"]
    if "dead" in F:
        e[F["dead"], :2] = -1e9
    c, d = p["team_cost"], p["dead_cost"]
    cost = np.array([[0, c, d], [c, 0, d], [d, d, 0]])
    pred = np.full(len(e), "X", "<U1")
    for a, b in HALVES:
        pred[a:b + 1] = STATES[viterbi(e[a:b + 1], cost)]
    raw = np.where(graded_mask(len(e)), STATES[e.argmax(1)], "X")
    team = lambda x: x[:, 0] - x[:, 1]
    dead = lambda x: x[:, 2] - np.logaddexp(x[:, 0], x[:, 1])
    lanes = [(name, kind, f(x) - f(np.zeros((1, 3)))) for kind, f in (("team", team), ("dead", dead))
             for name, x in cs.items()]
    lanes = [lane for lane in lanes if np.abs(lane[2]).max() > 0.05]  # drop clues that never speak to it
    lanes += [("all clues x weight", kind, f(e)) for kind, f in (("team", team), ("dead", dead))]
    return dict(pred=pred, raw=raw, lanes=lanes)
