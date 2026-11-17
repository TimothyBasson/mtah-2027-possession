"""PathCRF-style switch gating: a direct change of team is dearer, except at a detected touch by
the team that takes over (possession passes only through a reception, never "teleports").

    python alldata/oss/gate.py --tag pw1e --kind val      # pick the setting
    python alldata/oss/gate.py --tag pw1e --kind test --hi 6 --b 4 --win 6

Base is the shipped decode: touch evidence (weight 1, hold 3 s, threshold 0.5) on the all-data
net's held-out probabilities. The gate changes only the H<->A switch cost: `hi` everywhere, less
b x score within `win` frames of a touch by the incoming team.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "touch"))
import common as C  # noqa: E402
import tuse  # noqa: E402
from turnover import common as TC  # noqa: E402
import decode  # noqa: E402

EV = dict(thr=0.5, w=1.0, hold=3.0)


def decode_gate(probs, spans, n, ev, bonus, hi, cost=4, min_team=1, min_dead=1):
    out = np.full(n, C.X, np.int64)
    for p, (a, b) in zip(probs, spans):
        lp = np.log(np.clip(decode.upsample(p, b - a + 1), 1e-6, 1)) + ev[a:b + 1]
        path = TC.viterbi_ev(lp, cost, None if bonus is None else bonus[a:b + 1], ha_cost=hi)
        path = TC.absorb(path, min_team, min_dead)
        out[a:b + 1] = np.array([C.H, C.A, C.DEAD])[path]
    return out


def gate_bonus(n, f, tm, sc, b, win):
    """(n, 2) [to home, to away]: b x score within win frames of a touch by that team."""
    g = np.zeros((n, 2))
    for j in range(len(f)):
        lo, up = max(f[j] - win, 0), min(f[j] + win + 1, n)
        g[lo:up, tm[j]] = np.maximum(g[lo:up, tm[j]], b * sc[j])
    return g


def one(args):
    fold, name, tag, touch, settings = args
    raw, fo, spans = C.truths(name)
    n = len(raw)
    z = np.load(os.path.join(C.RUNS, tag, fold, name + ".npz"))
    pr = [z[f"s{i}"] for i in range(len(spans))]
    pf = [z[f"fs{i}"] for i in range(len(spans))]
    q = tuse.touch_probs(touch, fold, name, 1, n, spans)
    f, tm, sc = tuse.events(q, EV["thr"])
    ev, _ = tuse.evidence(n, f, tm, sc, EV["w"], EV["hold"])
    out = {"noev": C.evaluate(name, C.decode_ev(pr, spans, n, **C.RAW), C.decode_ev(pf, spans, n, **C.FOLDED)),
           "base": C.evaluate(name, C.decode_ev(pr, spans, n, ev, **C.RAW), C.decode_ev(pf, spans, n, ev, **C.FOLDED))}
    for hi, b, win in settings:
        g = gate_bonus(n, f, tm, sc, b, win)
        out[(hi, b, win)] = C.evaluate(name, decode_gate(pr, spans, n, ev, g, hi, **C.RAW),
                                       decode_gate(pf, spans, n, ev, g, hi, **C.FOLDED))
    return (fold, name), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--kind", default="val")
    ap.add_argument("--hi", nargs="+", type=float, default=[4, 6, 8, 12])
    ap.add_argument("--b", nargs="+", type=float, default=[2, 4, 8])
    ap.add_argument("--win", nargs="+", type=int, default=[3, 8])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    settings = [(h, b, w) for h in a.hi for b in a.b for w in a.win if b <= h + 4]
    ms = tuse.matches(a.tag, a.kind)
    res = dict(C.pool_map(one, [(fo, n, a.tag, a.touch, settings) for fo, n in ms], a.workers))
    R = {f"{n}@{fo}": r for (fo, n), r in res.items()}
    base = {k: r["base"] for k, r in R.items()}
    print(f"== {a.tag} {a.kind}: switch gating at touches ({a.touch}); deltas vs shipped decode (touch evidence)")
    C.report("no evidence", {k: r["noev"] for k, r in R.items()}, base)
    C.report("base: touch evidence", base)
    for s in settings:
        C.report(f"hi{s[0]:g} b{s[1]:g} win{s[2]}", {k: r[s] for k, r in R.items()}, base)


if __name__ == "__main__":
    main()
