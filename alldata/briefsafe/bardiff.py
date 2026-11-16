"""Label-free diff of two BAR-ATH files, per half: where the changed frames are, and whether the
new short possessions (under 2 s) of the second file are backed by a kink of BAR-ATH's ball.

    python alldata/briefsafe/bardiff.py evaluation/results/alldata_learned_tim_touch.csv evaluation/results/alldata_briefsafe_touch.csv
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "possession"), os.path.join(ROOT, "alldata", "contest"), ROOT]
import load  # noqa: E402
import cfeats  # noqa: E402

A, B = (pd.read_csv(f).label.astype(str).values for f in sys.argv[1:3])
players, ball, team, halves = load.load()
seen = np.isfinite(ball[:, 0])
z = np.load(os.path.join(ROOT, "data", "cache", "alldata", "contest_bar", "BAR-ATH.npz"))
names = [str(x) for x in z["names"]]
km = names.index("k_last_m")


def spells(x, a, b):
    e = np.flatnonzero(x[a + 1:b + 1] != x[a:b]) + a + 1
    s, t = np.r_[a, e], np.r_[e, b + 1]
    return [(i, j, x[i]) for i, j in zip(s, t) if x[i] in "HA"]


for hi, (a, b) in enumerate(halves):
    ch = A[a:b + 1] != B[a:b + 1]
    sa, sb = seen[a:b + 1], A[a:b + 1]
    print(f"half {hi + 1}: changed {ch.mean():.3f} of frames; of those ball seen {sa[ch].mean():.2f}, "
          f"first file D {(sb[ch] == 'D').mean():.2f}, second file D {(B[a:b + 1][ch] == 'D').mean():.2f}")
    C = z[f"F0_{hi}"][:, km]
    old = {(i, j, k) for i, j, k in spells(A, a, b)}
    new = [(i, j, k) for i, j, k in spells(B, a, b) if (i, j, k) not in old and (j - i) / 25 < 2]
    back = []
    for i, j, k in new:
        r0, r1 = (i - a) // cfeats.STEP, max((j - a) // cfeats.STEP, (i - a) // cfeats.STEP + 1)
        sg = 1.0 if k == "H" else -1.0
        back.append((C[r0:r1 + 1] * sg >= 0.1).any())
    nsa = sum((j - i) / 25 < 2 for i, j, _ in spells(A, a, b)); nsb = sum((j - i) / 25 < 2 for i, j, _ in spells(B, a, b))
    print(f"         under-2 s possessions {nsa} -> {nsb}; new ones {len(new)}, kink-backed (>=0.5 m) {np.mean(back) if back else float('nan'):.2f}")
