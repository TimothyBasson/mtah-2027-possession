"""Tracking around frames where the keeper clue changes v9 away from the notes."""
import csv, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hmm import decode as dec, keeper
bx, by, usable, px, py, ks, pitch, halves = dec.keeper_positions()
c = keeper.keeper_category(bx, by, usable, px, py, ks, pitch, 25, halves, use=(0, 1), ends="box")
L = lambda p: np.array([r["label"] for r in csv.DictReader(open(p))])
base, new = L("evaluation/results/kp_base.csv"), L("evaluation/results/kp_box8.csv")
import xml.etree.ElementTree as ET
for f0 in map(int, sys.argv[1:]):
    print("==", f0, "%d:%04.1f" % divmod(f0 / 25, 60))
    for g in range(f0 - 200, f0 + 300, 25):
        kd = []
        for s in (0, 1):
            k = ks[g, s]
            kd.append(np.hypot(px[g, k] - bx[g], py[g, k] - by[g]) if k >= 0 else np.nan)
        print("  %6d %s>%s cat %2d ball %-12s keeperd H %.1f A %.1f  kx %s" % (
            g, base[g], new[g], c[g], "-" if not usable[g] else "(%.0f,%.0f)" % (bx[g], by[g]),
            kd[0], kd[1], ",".join("%.0f" % px[g, ks[g, s]] if ks[g, s] >= 0 else "-" for s in (0, 1))))
