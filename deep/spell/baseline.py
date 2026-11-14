"""The fold nets as they are, under fixed decodes, on both label views: what a head has to beat.

    python deep/spell/baseline.py [--tag v2]

Reads the probabilities deep/train.py saved (data/cache/deep/<tag>/<fold>/). No decode setting is
picked per fold. "chain" is crf.py's minimum-duration chain with one flat, unlearned cost: the
decode a trained head uses, before any training through it.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from crf import MinDurCRF  # noqa: E402
from decode import labels_from  # noqa: E402
from head import decode_chain, decode_frames  # noqa: E402


def flat_chain(mins, cost):
    c = MinDurCRF(mins, init_cost=cost)
    return lambda Ps: decode_chain(c, [(5 * np.log(np.clip(P, 1e-6, 1))).astype(np.float32) for P in Ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--kind", default="test")
    a = ap.parse_args()
    by = {m["name"]: m for m in C.labels_all()}
    ways = {"shipped decode at 25 fps (cost 4, 1 s / 1 s)": None}
    for mt, md in ((1, 1), (2, 1), (2, 2), (2, 3)):
        ways[f"5 fps, cost 4, runs under {mt} s / {md} s absorbed"] = (lambda Ps, mt=mt, md=md: [decode_frames(P, 4, mt, md) for P in Ps])
    for mins in ((5, 5, 5), (10, 10, 5), (10, 10, 10), (10, 10, 15)):
        for cost in (2, 4, 8):
            ways[f"chain, floor {mins[0] / 5:g} s / {mins[2] / 5:g} s, flat cost {cost}"] = flat_chain(mins, cost)
    rows = {w: {} for w in ways}
    for fold in C.FOLDS:
        acc = {w: dict(raw=[], control=[]) for w in ways}
        for f in sorted(glob.glob(os.path.join(C.ROOT, "data", "cache", "deep", a.tag, fold, "*.npz"))):
            z = np.load(f)
            if str(z["kind"]) != a.kind:
                continue
            m = by[os.path.basename(f)[:-4]]
            Ps = [z[f"p{i}"] for i in range(len(m["spans"]))]
            for w, fn in ways.items():
                if fn is None:
                    lab = labels_from(Ps, m["spans"], len(m["labels"]["raw"]), cost=4, min_team=1, min_dead=1)
                    for view in acc[w]:
                        gt, pr = C.pm.label_codes(m["labels"][view]), C.pm.label_codes(lab)
                        runs = C.pm.valid_runs(gt)
                        acc[w][view].append(C.pm.spell_scores(gt, C.pm.fill_abstentions(pr, runs), runs)["mean_f1"])
                else:
                    paths = fn(Ps)
                    for view in acc[w]:
                        acc[w][view].append(C.spell_f1(m, paths, view))
        for w in ways:
            rows[w][fold] = {v: np.mean(s) for v, s in acc[w].items()}
    print(f"{'':44s} raw: idsse metrica skillcorner | mean   control: same")
    for w in ways:
        C.report(w[:44], rows[w])


if __name__ == "__main__":
    main()
