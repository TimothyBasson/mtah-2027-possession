"""Where true possessions are lost, by length and by what the prediction did instead.

    python deep/short/diag.py bt,bts1 [--teach-tag tb]

Held-out matches, probabilities averaged over the tags, shipped decode. For every true possession
not matched at IoU 0.5: was the model's mean probability of the true team under 0.5 inside it
("blind": the turnover is not in the probabilities), or over 0.5 but the decode drew it wrong
("seen, lost in decode")? Raw labels and labels folded to the control rule.
"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode, score_hmm as sh
from control import control_labels
from run import DECODE
from bylength import runs

BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]

def one(args):
    path, tags = args
    name = os.path.basename(path)[:-4]
    d = sh.load(path)
    spans = [tuple(h) for h in d["halves"]]
    ps = [np.load(os.path.join(ROOT, "data", "cache", "deep", t, name.split("_")[0], name + ".npz")) for t in tags]
    probs = [np.mean([p[f"p{i}"] for p in ps], 0) for i in range(len(spans))]
    pred = decode.labels_from(probs, spans, len(d["labels"]), **DECODE)
    pred[d["labels"] == "X"] = "X"
    full = np.full((len(d["labels"]), 3), np.nan)
    for p, (a, b) in zip(probs, spans):
        full[a:b + 1] = decode.upsample(p, b - a + 1)
    out = []
    for ti, truth in enumerate((d["labels"], control_labels(d))):
        pr = runs(pred)
        for k, a, b in runs(truth):
            best = 0
            for pk, pa, pb in pr:
                if pb <= a or pa >= b or pk != k: continue
                best = max(best, (min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa)))
            c = "HA".index(k)
            seg = full[a:b]
            pt = np.nanmean(seg[:, c]); po = np.nanmean(seg[:, 1 - c]); pdd = np.nanmean(seg[:, 2])
            # the strongest 0.4 s the true team gets inside it
            peak = np.nanmax(seg[:, c])
            prev = truth[a - 1] if a else "X"; nxt = truth[b] if b < len(truth) else "X"
            out.append((ti, (b - a) / 25, best, pt, po, pdd, peak, "HADX".index(prev), "HADX".index(nxt), c))
    return name, np.array(out)

def main():
    tags = sys.argv[1].split(",")
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    with Pool(2) as pool:
        rows = pool.map(one, [(p, tags) for p in paths])
    for ti, tn in enumerate(("labels as they are", "folded labels")):
        print(f"\n== {tn}: true possessions by length; found at IoU 0.5; of the missed, mean p(true team) / p(other) / p(dead); share blind (p true < p other or dead)")
        for prov in ("idsse", "metrica", "skillcorner"):
            R = np.concatenate([r for n, r in rows if n.startswith(prov)]); R = R[R[:, 0] == ti]
            print(f" {prov}: {len(R)} possessions")
            for lo, hi in BINS:
                m = (R[:, 1] >= lo) & (R[:, 1] < hi); S = R[m]; miss = S[S[:, 2] < 0.5]
                blind = (miss[:, 3] < np.maximum(miss[:, 4], miss[:, 5]))
                other = (miss[:, 4] > np.maximum(miss[:, 3], miss[:, 5])); dead = (miss[:, 5] > np.maximum(miss[:, 3], miss[:, 4]))
                print(f"   {lo:>2}-{hi if hi < 100 else 'inf':<4} share {m.mean():5.1%}  found {1 - len(miss) / max(len(S), 1):5.1%}  "
                      f"missed: p {miss[:, 3].mean():.2f}/{miss[:, 4].mean():.2f}/{miss[:, 5].mean():.2f}  "
                      f"called other {other.mean():5.1%} dead {dead.mean():5.1%} seen-but-lost {1 - blind.mean():5.1%}  peak>0.5 {np.mean(miss[:, 6] > 0.5):5.1%}")
    np.save(os.path.join(ROOT, "data", "cache", "diag_" + "_".join(tags) + ".npy"), np.concatenate([r for n, r in rows]))

if __name__ == "__main__":
    main()
