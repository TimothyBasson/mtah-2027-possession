"""Late blends of saved held-out probabilities from different model families, at score.py's fixed
decodes, on labels as built and folded.

    python alldata/learned/blend.py --base pw1e --other phys:sLh --weights 0.25 0.5

Every family must have been shown the same simulated ball (build.ball_version 1, key "s").
--other takes run tags of this folder, "deep:<tag>", "phys:<run>" (alldata/physics predictions made
on this folder's balls) or "tim" (tim_barlike.py's decoded labels and classifier posteriors, v1).
The blend is the weighted mean of probabilities; weights are fixed in advance, not fitted.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
from score import pm, decode  # noqa: E402

CACHE = os.path.join(S.ROOT, "data", "cache", "alldata")


def load(tag, fold, name, pre, nh, spans):
    """List of (T, 3) per half for one source; None if missing."""
    if tag.startswith("phys:"):
        z = np.load(os.path.join(CACHE, "pred", tag[5:], name + ".npz"))
        return [z[f"p{i}"] for i in range(nh)]
    if tag in ("tim", "timlab"):  # 25 fps arrays over the whole match -> this folder's 5 fps rows
        z = np.load(os.path.join(CACHE, "tim", name + ".v1.npz"))
        if tag == "tim":
            full = z["post"]
        else:
            full = np.full((len(z["labels"]), 3), 0.05, np.float32)
            for k, c in enumerate("HAD"):
                full[z["labels"] == c, k] = 0.9
        return [full[a:b + 1:5] for a, b in spans]
    z = np.load(os.path.join(S.tag_dir(tag), fold, name + ".npz"))
    k = "s" if "s0" in z.files else "p"
    kk = pre + k if f"{pre}{k}0" in z.files else k
    return [z[f"{kk}{i}"] for i in range(nh)]


def one(args):
    fold, name, base, others, weights = args
    raw, fl, spans = S.truths(name)
    row = []
    for t, dec, pre in ((raw, S.RAW, ""), (fl, S.FOLDED, "f")):
        b = load(base, fold, name, pre, len(spans), spans)
        srcs = [load(o, fold, name, pre, len(spans), spans) for o in others]
        cands = [b] + srcs
        for w in weights:
            mix = []
            for h in range(len(spans)):
                n = min(len(x[h]) for x in cands)
                m = (1 - w) * b[h][:n] + w * np.mean([x[h][:n] for x in srcs], 0)
                mix.append(m / m.sum(1, keepdims=True))
            cands.append(mix)
        for c in cands:
            p = pm.label_codes(decode.labels_from(c, spans, len(t), **dec))
            row.append(pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"])
    return name, row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.5])
    a = ap.parse_args()
    jobs = []
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(a.base), fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                jobs.append((fold, os.path.basename(f)[:-4], a.base, a.other, a.weights))
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(one, jobs))
    names = [a.base] + a.other + [f"blend w={w}" for w in a.weights]
    k = len(names)
    R = np.array([res[j[1]] for j in jobs])
    provs = ["idsse", "metrica", "skillcorner"]
    for off, title in ((0, "labels as built"), (k, "folded labels")):
        print(title)
        for i, nm in enumerate(names):
            per = [R[[x for x, j in enumerate(jobs) if j[0] == p], off + i].mean() for p in provs]
            d = R[:, off + i] - R[:, off]
            print(f"  {nm:16s}" + "".join(f"{p[:8]:>9s} {x:.3f}" for p, x in zip(provs, per)) + f"   mean {np.mean(per):.3f}"
                  + (f"   vs base {d.mean():+.3f} (se {d.std(ddof=1) / np.sqrt(len(d)):.3f}), up {(d > 0).sum()}/{len(d)}" if i else ""))
