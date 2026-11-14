"""deep/'s held-out matches against hmm v7's, match by match, on the outside labels as they are
and folded towards the challenge's controlled-possession rule (external/control.py).

    python external/control_features.py && python external/train_control.py --loso   # once, for hmm
    python deep/folded.py [--tag v2] [--jobs 2]

deep: the fold net that never saw the match's provider, the shipped decode in every fold.
hmm v7: exactly external/score_control.py's run() at the shipped weight (classifier trained
without the provider scored, tables counted on the others, hand-set weights as shipped, ball at
59% seen). Both are scored by external/score_hmm.py's score(). Prints each provider's mean, the
mean of the three providers, the mean of all 30 matches, and deep minus hmm paired by match.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import score_control as sc  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from run import DECODE  # noqa: E402

HMM = "v7 w{:g}".format(sc.dec.CONTROL_WEIGHT)


def one(args):
    name, tag = args
    d = sh.load(sc.STATE["paths"][name])
    p = np.load(os.path.join(ROOT, "data", "cache", "deep", tag, name.split("_")[0], name + ".npz"))
    spans = [tuple(h) for h in d["halves"]]
    pred = decode.labels_from([p[f"p{i}"] for i in range(len(spans))], spans, len(d["labels"]), **DECODE)
    pred[d["labels"] == "X"] = "X"
    views = (d["labels"], control_labels(d, sc.STATE["fold_seconds"]))
    return name, [sh.score(gt, pred)[0] for gt in views], [v[0] for v in sc.run(name)[1][HMM]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    state = dict(paths=paths, tables=sc.tables(paths), loso=dict(np.load(os.path.join(sc.CACHE, "control_loso.npz"))),
                 weights=[sc.dec.CONTROL_WEIGHT], fold_seconds=sc.CONTROL_SECONDS)
    with Pool(a.jobs, initializer=sc._init, initargs=(state,)) as pool:
        rows = pool.map(one, [(n, a.tag) for n in paths])
    for v, view in enumerate(("labels as they are", "folded labels")):
        print(f"\n{view}: spellF1        deep     hmm   deep - hmm (standard error, matches up)")
        means = []
        for k in list(sh.SETS) + ["all 30 matches"]:
            r = [(dp[v], hm[v]) for n, dp, hm in rows if n.startswith(k) or k == "all 30 matches"]
            dp, hm = np.array(r).T
            diff = dp - hm
            if k != "all 30 matches":
                means.append((dp.mean(), hm.mean()))
            print(f"  {k + ' (' + str(len(r)) + ')' if k != 'all 30 matches' else k:28s} {dp.mean():.3f}   {hm.mean():.3f}   "
                  f"{diff.mean():+.3f} (se {diff.std(ddof=1) / np.sqrt(len(diff)):.3f}, up {int((diff > 0).sum())}/{len(diff)})")
        m = np.mean(means, 0)
        print(f"  {'mean of the three providers':28s} {m[0]:.3f}   {m[1]:.3f}   {m[0] - m[1]:+.3f}")
    print("\nper match, raw then folded: deep, hmm")
    for n, dp, hm in rows:
        print(f"  {n:22s} {dp[0]:.3f} {hm[0]:.3f}   {dp[1]:.3f} {hm[1]:.3f}")


if __name__ == "__main__":
    main()
