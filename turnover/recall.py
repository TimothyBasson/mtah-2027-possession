"""How many true turnovers does a set of labels find (same direction, within tol seconds), by the
length of the shorter possession beside the turnover."""
import sys
from multiprocessing import Pool
import numpy as np
import common as C
from common import sh

TAGS = ["bt", "bts1"]
BINS = [(0, 1), (1, 2), (2, 5), (5, 1e9)]

def match_events(tf, td, pf, pd_, tol):
    """greedy one-to-one, same direction; returns hit flags for truth, and for pred"""
    ht, hp = np.zeros(len(tf), bool), np.zeros(len(pf), bool)
    j0 = 0
    for i, (f, dr) in enumerate(zip(tf, td)):
        best, bd = -1, tol + 1
        for j in range(max(0, np.searchsorted(pf, f - tol)), len(pf)):
            if pf[j] > f + tol: break
            if not hp[j] and pd_[j] == dr and abs(pf[j] - f) < bd:
                best, bd = j, abs(pf[j] - f)
        if best >= 0:
            ht[i] = hp[best] = True
    return ht, hp

def shortside(gt, tv):
    n = len(gt)
    ch = np.r_[0, np.flatnonzero(gt[1:] != gt[:-1]) + 1, n]
    k = np.searchsorted(ch, tv)
    return np.minimum(ch[k + 1] - ch[k], ch[k] - ch[k - 1]) / 25

def one(path):
    name = path.split("/")[-1][:-4]
    d = sh.load(path)
    spans = [tuple(h) for h in d["halves"]]
    n = len(d["labels"])
    pred = C.labels_ev(C.base_probs(name, TAGS, len(spans)), spans, n)
    pred[d["labels"] == "X"] = "X"
    pf = C.turnovers(pred); pdir = pred[pf]
    out = []
    for gt in (d["labels"], C.control_labels(d)):
        tv = C.turnovers(gt)
        row = []
        for tol in (12, 25):
            ht, hp = match_events(tv, gt[tv], pf, pdir, tol)
            L = shortside(gt, tv)
            row.append([[ht[(L >= lo) & (L < hi)].sum(), ((L >= lo) & (L < hi)).sum()] for lo, hi in BINS] + [[hp.sum(), len(hp)]])
        out.append(row)
    return name, np.array(out)

if __name__ == "__main__":
    with Pool(4) as pool:
        rows = pool.map(one, C.paths())
    names = [r[0] for r in rows]; S = np.stack([r[1] for r in rows])  # match, truth, tol, bin, (hit, n)
    for ti, tn in enumerate(("labels as given", "folded labels")):
        for oi, tol in enumerate((0.5, 1.0)):
            print(f"\n{tn}, within {tol} s: recall by shorter neighbouring possession <1s, 1-2s, 2-5s, >5s (count/match) | precision")
            for p in C.PROVIDERS:
                m = np.array([x.startswith(p) for x in names]); s = S[m, ti, oi].sum(0)
                print(f"  {p:12s}" + "  ".join(f"{h / max(k,1):.2f} ({k / m.sum():.0f})" for h, k in s[:-1]) + f"  | all {s[:-1,0].sum()/s[:-1,1].sum():.2f} | prec {s[-1,0] / s[-1,1]:.2f} ({s[-1,1]/m.sum():.0f} predicted)")
