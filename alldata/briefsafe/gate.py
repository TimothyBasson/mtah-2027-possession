"""Kink gate: take the fine-cut run (B) only where the ball backs it, the base run (A) elsewhere.

Where A and B name different teams, B's team is kept only if the last kink of the shown ball
(cfeats.py: where the ball was really played) was nearer that team by at least --m (in the
column's units, 1 = 5 m); otherwise A's probabilities are used. No labels, no fitting, the same
rule on every provider.

    python alldata/briefsafe/gate.py pw1e pcr1e gk [--m 0.1] [--mode last|either]
"""
import argparse, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "data", "cache", "alldata", "runs")
CON = os.path.join(ROOT, "data", "cache", "alldata", "contest")
EVAL = {0: "p", 1: "s"}

ap = argparse.ArgumentParser()
ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("out")
ap.add_argument("--m", type=float, default=0.1)
ap.add_argument("--mode", default="last", choices=["last", "either", "run", "org", "lastorg"])
ap.add_argument("--con", default=CON)
a = ap.parse_args()


def gate(pa, pb, C, names):
    ta, tb = pa.argmax(1), pb.argmax(1)
    diff = (ta != tb) & (tb < 2)
    sgn = np.where(tb == 0, 1.0, -1.0)
    m = C[:, names.index("k_last_m")] * sgn
    ok = m >= a.m
    if a.mode == "either":
        ok |= C[:, names.index("k_next_m")] * sgn >= a.m
    if a.mode == "org":       # who played the ball now in flight, instead of the last kink
        ok = C[:, names.index("o_m")] * sgn >= a.m
    if a.mode == "lastorg":   # last kink, and the ball's line back does not point at the other team
        ok &= C[:, names.index("o_m")] * sgn > -a.m
    if a.mode == "run":
        ok |= C[:, names.index("k_last_run")] * sgn > 0
    use_a = diff & ~ok
    out = pb.copy()
    out[use_a] = pa[use_a]
    return out, use_a.mean(), diff.mean()


stats = []
for f in sorted(glob.glob(os.path.join(RUNS, a.A, "*", "*.npz"))):
    fold, fn = f.split(os.sep)[-2:]
    g = os.path.join(RUNS, a.B, fold, fn)
    if not os.path.exists(g):
        continue
    za, zb = np.load(f), np.load(g)
    c = np.load(os.path.join(a.con, fn))
    names = [str(x) for x in c["names"]]
    o = {"kind": za["kind"]}
    for k in za.files:
        if k == "kind":
            continue
        key = k[1:] if k.startswith("f") else k
        v = {vv: kk for vv, kk in EVAL.items()}
        ver = [vv for vv, kk in EVAL.items() if key.startswith(kk)][0]
        i = int(key[1:])
        C = c[f"F{ver}_{i}"]
        n = min(len(C), len(za[k]))
        o[k], ua, df = gate(za[k][:n], zb[k][:n], C[:n], names)
        if len(za[k]) > n:
            o[k] = np.concatenate([o[k], zb[k][n:]])
        if k == "s0":
            stats.append((ua, df))
    os.makedirs(os.path.join(RUNS, a.out, fold), exist_ok=True)
    np.savez_compressed(os.path.join(RUNS, a.out, fold, fn), **o)
s = np.array(stats)
print(f"{a.out}: rows where the two disagree {s[:, 1].mean():.3f}, of which sent back to {a.A} {s[:, 0].mean() / max(s[:, 1].mean(), 1e-9):.2f}")
