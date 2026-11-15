"""Share of true possessions found (IoU 0.5 with a predicted possession of the same team), by
length, for one or more tags' held-out matches, at one fixed decode.

    python deep/bylength.py v2 bt
"""

import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import decode  # noqa: E402

FIXED = dict(cost=8, min_team=1, min_dead=1)
BINS = [(0, 2), (2, 5), (5, 10), (10, 1e9)]


def runs(lab):
    out, i = [], 0
    for k, g in itertools.groupby(lab):
        n = len(list(g))
        if k in "HA":
            out.append((k, i, i + n))
        i += n
    return out


def found(truth, pred):
    tr, pr = runs(truth), runs(pred)
    hit, j0 = [], 0
    for k, a, b in tr:
        while j0 < len(pr) and pr[j0][2] <= a:
            j0 += 1
        ok, j = False, j0
        while j < len(pr) and pr[j][1] < b:
            pk, pa, pb = pr[j]
            inter = min(b, pb) - max(a, pa)
            if pk == k and inter / (max(b, pb) - min(a, pa)) >= 0.5:
                ok = True
            j += 1
        hit.append(((b - a) / 25, ok))
    return hit, len(pr), len(tr)


def main():
    for tag in sys.argv[1:]:
        print(tag)
        for fold in ("idsse", "metrica", "skillcorner"):
            hits, npred, ntrue = [], 0, 0
            for f in sorted(glob.glob(os.path.join(D.ROOT, "data", "cache", "deep", tag, fold, "*.npz"))):
                z = np.load(f)
                if str(z["kind"]) != "test":
                    continue
                d = np.load(os.path.join(D.ROOT, "data", "external", os.path.basename(f)))
                spans = [tuple(h) for h in d["halves"]]
                pred = decode.labels_from([z[f"p{i}"] for i in range(len(spans))], spans, len(d["labels"]), **FIXED)
                pred[d["labels"] == "X"] = "X"
                h, a, b = found(d["labels"], pred)
                hits += h; npred += a; ntrue += b
            L = np.array([x[0] for x in hits]); ok = np.array([x[1] for x in hits])
            print(f"  {fold:12s}" + "".join(f"  {100 * ok[(L >= lo) & (L < hi)].mean():3.0f}%" for lo, hi in BINS)
                  + f"   {npred} / {ntrue}")


if __name__ == "__main__":
    main()
