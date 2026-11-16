"""Brief possessions written into the shipped decode where the other team's probability peaks.

    python alldata/shortdec/insert.py --tag pw1e            # held out by provider
    python alldata/shortdec/insert.py --bar                 # the public 5 minutes of BAR-ATH (a check)

The evidence is one head of the net (labels as built, folded, or both averaged); the plain decode
is always the shipped one on the averaged heads.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import dec  # noqa: E402
from common import pm  # noqa: E402

GRID = [dict(tau=1.0)] + [dict(head=h, tau=t, ins_min=m, rho=r)
                          for h in ("built", "both") for t in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
                          for m in (0.4,) for r in (0.5,)]


def one(args):
    fold, name, f, key, combos = args
    probs, spans, raw, fd = C.load(f, key)
    heads = {h: [dec.upsample(p, b - a + 1) for p, (a, b) in zip(C.load(f, key, h)[0], spans)] for h in ("both", "built", "folded")}
    truths = [(t, pm.valid_runs(t)) for t in (raw, fd)]
    out = []
    for c in combos:
        c = dict(c); h = c.pop("head", "both")
        pred = np.full(len(raw), 3, np.int8)
        for P, Pe, (a, b) in zip(heads["both"], heads[h], spans):
            pred[a:b + 1] = dec.decode_insert(P, Pe, **c)
        row = []
        for t, runs in truths:
            f1, st = C.slice_stats(t, pm.fill_abstentions(pred, runs), runs)
            row.append(np.r_[f1, st])
        out.append(row)
    return np.array(out)


def run(tag, combos, key="s", kind="test"):
    ms = C.matches(tag, kind)
    with ProcessPoolExecutor(4) as ex:
        R = np.array(list(ex.map(one, [(fo, n, f, key, combos) for fo, n, f in ms])))
    return ms, R


def fmt(ms, R, ci):
    names = [n for _, n, _ in ms]
    out = []
    for li in range(2):
        f = C.provider_mean(names, R[:, ci, li, 0])
        s = C.provider_sum(names, R[:, ci, li, 1:]).sum(0)
        out.append(" ".join(f"{x:.3f}" for x in f) + f" | brief {s[1] / s[0]:.2f} nb {s[3] / s[2]:.2f} merged {s[4] / s[0]:.2f} drawn/true {s[6] / s[5]:.2f}")
    return "  ||  ".join(out)


def bar(combos):
    import pandas as pd
    sys.path += [os.path.join(C.ROOT, "possession"), os.path.join(C.ROOT, "external"), C.ROOT]
    import load
    _, _, _, halves = load.load()
    t = pm.label_codes(pd.read_csv(os.path.join(C.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))).astype(np.int8)
    runs = pm.valid_runs(t)
    Z = {h: np.load(os.path.join(C.ROOT, "data", "cache", "alldata", "bar_ath", h + ".npz")) for h in ("both", "built", "folded")}
    heads = {h: [dec.upsample(Z[h][f"p{i}"], b - a + 1) for i, (a, b) in enumerate(halves)] for h in Z}
    for c in combos:
        cc = dict(c); h = cc.pop("head", "both")
        pred = np.full(len(t), 3, np.int8)
        for P, Pe, (a, b) in zip(heads["both"], heads[h], halves):
            pred[a:b + 1] = dec.decode_insert(P, Pe, **cc)
        f1, g, p, fl = C.spell_f1(t, pm.fill_abstentions(pred, runs), runs, detail=True)
        n = [len(C.spells(pred[a:b + 1], [(0, b - a + 1)])) for a, b in halves]
        print(f"public window {f1:.3f}  found {int(fl[2].sum())}/{len(g)}  drawn in window {len(p)}  possessions per half {n}", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--key", default="s")
    ap.add_argument("--bar", action="store_true")
    a = ap.parse_args()
    if a.bar:
        return bar(GRID)
    ms, R = run(a.tag, GRID, a.key)
    print("columns: IDSSE Metrica SkillCorner mean, labels as built || folded")
    for ci, c in enumerate(GRID):
        print(fmt(ms, R, ci), c)


if __name__ == "__main__":
    main()
