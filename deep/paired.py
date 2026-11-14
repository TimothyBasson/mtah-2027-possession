"""Two sets of tags against each other, match by match, on the outside labels as they are and
folded towards the controlled-possession rule (external/control.py), at the shipped decode and
at a 2 s minimum possession.

    python deep/paired.py v2,v2s1 bt,bts1

Each match is scored with the fold net that never saw its provider; a set's score for a match is
the mean over its tags (seeds). The folded truth is built from the scored match's own ball: that
is the yardstick, never the prediction.
"""

import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from run import DECODE  # noqa: E402

DECODES = (("shipped decode", DECODE), ("2 s minimum possession", dict(DECODE, min_team=2)))


def one(args):
    path, tags = args
    name = os.path.basename(path)[:-4]
    d = sh.load(path)
    spans = [tuple(h) for h in d["halves"]]
    truths = (d["labels"], control_labels(d))
    out = []
    for tag in tags:
        p = np.load(os.path.join(ROOT, "data", "cache", "deep", tag, name.split("_")[0], name + ".npz"))
        row = []
        for _, dec in DECODES:
            pred = decode.labels_from([p[f"p{i}"] for i in range(len(spans))], spans, len(d["labels"]), **dec)
            pred[d["labels"] == "X"] = "X"
            row.append([sh.score(gt, pred)[0] for gt in truths])
        out.append(row)
    return name, np.array(out)  # (tags, decodes, truths)


def main():
    sets = [s.split(",") for s in sys.argv[1:3]]
    tags = sets[0] + sets[1]
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    with Pool(4) as pool:
        rows = pool.map(one, [(p, tags) for p in paths])
    names = [n for n, _ in rows]
    S = np.stack([r for _, r in rows])  # (matches, tags, decodes, truths)
    A, B = S[:, :len(sets[0])].mean(1), S[:, len(sets[0]):].mean(1)
    for di, (dn, _) in enumerate(DECODES):
        for ti, tn in enumerate(("labels as they are", "folded labels")):
            print(f"\n{dn}, {tn}: spellF1   {sys.argv[1]:>10s} {sys.argv[2]:>10s}   second - first (standard error, matches up)")
            means = []
            for k in ("idsse", "metrica", "skillcorner", "all"):
                m = np.array([n.startswith(k) or k == "all" for n in names])
                a, b = A[m, di, ti], B[m, di, ti]
                diff = b - a
                if k != "all":
                    means.append((a.mean(), b.mean()))
                print(f"  {k:12s} ({m.sum():2d})  {a.mean():10.3f} {b.mean():10.3f}   {diff.mean():+.3f} "
                      f"(se {diff.std(ddof=1) / np.sqrt(len(diff)):.3f}, up {int((diff > 0).sum())}/{len(diff)})")
            mm = np.mean(means, 0)
            print(f"  {'provider mean':17s}  {mm[0]:10.3f} {mm[1]:10.3f}   {mm[1] - mm[0]:+.3f}")


if __name__ == "__main__":
    main()
