"""Learned switch costs (the PathCRF idea, cheaply): the net's change head lowers the H<->A
switch cost in the decode; elsewhere a switch costs `hi`.

    python alldata/oss/edge_dec.py --tag ew1 --kind val
    python alldata/oss/edge_dec.py --tag ew1 --kind test --hi 8 --b 6

Every row is on the same net's probabilities, so the control (`hi4 b0`) is that net decoded the
shipped way. `ev` rows add the shipped touch evidence (alldata/touch, weight 1, hold 3 s).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "touch")]
import common as C  # noqa: E402
import tuse  # noqa: E402
import decode  # noqa: E402
from gate import EV, decode_gate  # noqa: E402


def one(args):
    fold, name, tag, settings, touch = args
    raw, fo, spans = C.truths(name)
    n = len(raw)
    z = np.load(os.path.join(C.RUNS, tag, fold, name + ".npz"))
    pr = [z[f"s{i}"] for i in range(len(spans))]
    pf = [z[f"fs{i}"] for i in range(len(spans))]
    e = np.zeros(n)
    for i, (a, b) in enumerate(spans):
        e[a:b + 1] = decode.upsample(z[f"es{i}"][:, None], b - a + 1)[:, 0]
    q = tuse.touch_probs(touch, fold, name, 1, n, spans)
    f, tm, sc = tuse.events(q, EV["thr"])
    ev1, _ = tuse.evidence(n, f, tm, sc, EV["w"], EV["hold"])
    out = {}
    for hi, b, use_ev in settings:
        ev = ev1 if use_ev else np.zeros((n, 3))
        g = np.stack([b * e, b * e], 1)
        out[(hi, b, use_ev)] = C.evaluate(name, decode_gate(pr, spans, n, ev, g, hi, **C.RAW),
                                          decode_gate(pf, spans, n, ev, g, hi, **C.FOLDED))
    return (fold, name), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ew1")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--kind", default="val")
    ap.add_argument("--hi", nargs="+", type=float, default=[4, 6, 8, 12])
    ap.add_argument("--b", nargs="+", type=float, default=[0, 2, 4, 8])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    settings = [(4.0, 0.0, 0), (4.0, 0.0, 1)] + [(h, b, ev) for ev in (0, 1) for h in a.hi for b in a.b if (h, b) != (4, 0)]
    ms = tuse.matches(a.tag, a.kind)
    res = dict(C.pool_map(one, [(fo, n, a.tag, settings, a.touch) for fo, n in ms], a.workers))
    R = {f"{n}@{fo}": r for (fo, n), r in res.items()}
    print(f"== {a.tag} {a.kind}: change head as switch evidence; deltas vs the same net decoded the shipped way without touches")
    base = {k: r[(4.0, 0.0, 0)] for k, r in R.items()}
    base_ev = {k: r[(4.0, 0.0, 1)] for k, r in R.items()}
    for s in settings:
        C.report(f"{'ev ' if s[2] else ''}hi{s[0]:g} b{s[1]:g}", {k: r[s] for k, r in R.items()}, base_ev if s[2] else base)


if __name__ == "__main__":
    main()
