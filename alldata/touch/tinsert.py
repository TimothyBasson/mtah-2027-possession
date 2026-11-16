"""Detected touches as forced possessions: a rule on top of the decoded labels.

    python alldata/touch/tinsert.py --tag pw1e --touch t1 [--kind test] [--v 1]

Where the decode says team O has the ball and the detector sees a touch by team K (score over a
threshold) that is followed within `hold` seconds by a touch of O, the frames from K's touch to
O's next touch become a possession of K. That is the brief touch whose miss merges the two
possessions of O either side. Nothing else in the labels changes. --kind val for picking.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import tuse  # noqa: E402

CODE = np.array([C.H, C.A])


def insert(p, f, tm, sc, thr, hold, othr=0.3, mink=1, pk=None, pmin=0.0, min_len=0.0, fps=25):
    """p: decoded codes (n,), events (frames, teams, scores) -> new codes."""
    out = p.copy()
    for j in np.flatnonzero(sc >= thr):
        k, o = CODE[tm[j]], CODE[1 - tm[j]]
        a = f[j]
        if p[a] != o:
            continue
        nx = np.flatnonzero((f > a) & (tm != tm[j]) & (sc >= othr) & (f <= a + hold * fps))
        if not len(nx):
            continue
        b = f[nx[0]]
        if b - a < min_len * fps or not (p[a:b] == o).all():
            continue
        if mink > 1 and ((f >= a) & (f < b) & (tm == tm[j]) & (sc >= othr)).sum() < mink:
            continue  # the control test of the folded labels: the team touches it twice
        if pmin > 0 and pk[a:b, tm[j]].mean() < pmin:
            continue  # the net itself gives the touching team almost no chance here
        out[a:b] = k
    return out


def one(args):
    fold, name, tag, touch, v, settings = args
    raw, fo, spans = C.truths(name)
    n = len(raw)
    key = tuse.KEY[v]
    z = np.load(os.path.join(C.RUNS, tag, fold, name + ".npz"))
    nh = len(spans)
    pr = [z[f"{key}{i}"] for i in range(nh)]
    pf = [z[f"f{key}{i}"] for i in range(nh)]
    pb = [(x + y) / 2 for x, y in zip(pr, pf)]
    q = tuse.touch_probs(touch, fold, name, v, n, spans)
    f, tm, sc = tuse.events(q, 0.3)
    d_raw, d_fold, d_both = C.decode_ev(pr, spans, n, **C.RAW), C.decode_ev(pf, spans, n, **C.FOLDED), C.decode_ev(pb, spans, n, **C.RAW)
    P = []
    for pp in (pr, pf, pb):
        Q = np.full((n, 3), 1 / 3)
        for i, (a, b) in enumerate(spans):
            Q[a:b + 1] = C.decode.upsample(pp[i], b - a + 1)
        P.append(Q)
    out = {"base": C.evaluate(name, d_raw, d_fold), "base_file": C.evaluate(name, d_both, d_both)}
    for s in settings:
        thr, hold, mink, pmin = s
        out[s] = C.evaluate(name, insert(d_raw, f, tm, sc, thr, hold, 0.3, mink, P[0], pmin), insert(d_fold, f, tm, sc, thr, hold, 0.3, mink, P[1], pmin))
        ib = insert(d_both, f, tm, sc, thr, hold, 0.3, mink, P[2], pmin)
        out[s + ("file",)] = C.evaluate(name, ib, ib)
    return (fold, name), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--kind", default="test")
    ap.add_argument("--v", type=int, default=1)
    ap.add_argument("--thr", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--hold", nargs="+", type=float, default=[2, 3, 5])
    ap.add_argument("--mink", nargs="+", type=int, default=[1, 2], help="touches by the team inside the forced possession")
    ap.add_argument("--pmin", nargs="+", type=float, default=[0.0, 0.1, 0.25], help="least mean probability the net gives that team there")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    settings = [(t, h, k, pm_) for t in a.thr for h in a.hold for k in a.mink for pm_ in a.pmin]
    ms = tuse.matches(a.tag, a.kind)
    res = dict(C.pool_map(one, [(fo, n, a.tag, a.touch, a.v, settings) for fo, n in ms], a.workers))
    R = {f"{n}@{fo}": r for (fo, n), r in res.items()}
    print(f"== {a.tag} + forced possessions from touches {a.touch}, {a.kind}, ball version {a.v}; IDSSE Metrica SkillCorner | mean")
    print("-- each head at its own decode, against its own labels")
    base = {n: r["base"] for n, r in R.items()}
    C.report("base", base)
    for s in settings:
        C.report(f"thr{s[0]} hold{s[1]} n{s[2]} p{s[3]}", {n: r[s] for n, r in R.items()}, base)
    print("-- the file's decode (two heads averaged, 1 s minimum), one labelling against both truths")
    base = {n: r["base_file"] for n, r in R.items()}
    C.report("base", base)
    for s in settings:
        C.report(f"thr{s[0]} hold{s[1]} n{s[2]} p{s[3]}", {n: r[s + ("file",)] for n, r in R.items()}, base)


if __name__ == "__main__":
    main()
