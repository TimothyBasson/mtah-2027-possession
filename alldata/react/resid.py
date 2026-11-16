"""The team reactions as a correction to the base model's team call, starting from the base.

    python alldata/react/resid.py [--seeds 0 1] [--trees 100] [--weight-brief 1] [--target folded]

On live rows the base model's log-odds of Home against Away are the starting score of a
gradient-boosted model whose inputs are the reaction rows (and, with --with-base, the base's
probabilities up to 4 s either side). It can only add to the base's log-odds what the reactions
explain beyond it; the alive/dead split is left alone. With zero trees it is the base exactly.
New probabilities are decoded like the submission.

Held out by provider as stack.py: fitted on the other two providers' held-out matches.
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdata  # noqa: E402
import react as R  # noqa: E402
import stack as S  # noqa: E402
from common import PROVS  # noqa: E402


def odds(p):
    return np.log(np.clip(p[:, 0], 1e-4, 1)) - np.log(np.clip(p[:, 1], 1e-4, 1))


def inputs(m, i, with_base):
    r = m.R[i]
    if not with_base:
        return r, R.swap(r)
    B = S.base_cols(m.probs[i])
    if with_base == "only":  # the control: the same correction without the reaction rows
        return B, S.swap_base(B)
    return np.hstack([r, B]), np.hstack([R.swap(r), S.swap_base(B)])


def fit(ms, target, seed, trees, wb, with_base, lr, sub=2):
    X, y, w, init = [], [], [], []
    for m in ms:
        t = m.folded if target == "folded" else m.raw
        bm = S.brief_mask(t)
        for i, fr in enumerate(m.fr):
            lab = t[fr]
            k = np.flatnonzero(lab < 2)[seed % sub::sub]
            x, xs = inputs(m, i, with_base)
            o = odds(m.probs[i])[k]
            X += [x[k], xs[k]]
            y += [1 - lab[k], lab[k]]          # 1 = Home
            init += [o, -o]
            w += [np.where(bm[fr][k], wb, 1.0)] * 2
    ds = lgb.Dataset(np.concatenate(X), np.concatenate(y), weight=np.concatenate(w), init_score=np.concatenate(init))
    return lgb.train(dict(objective="binary", learning_rate=lr, num_leaves=31, min_data_in_leaf=300, bagging_fraction=0.7,
                          bagging_freq=1, feature_fraction=0.7, seed=seed, num_threads=4, verbose=-1), ds, trees)


def apply(boosters, m, with_base, scale=1.0):
    out = []
    for i, p in enumerate(m.probs):
        x, xs = inputs(m, i, with_base)
        d = np.mean([(b.predict(x, raw_score=True) - b.predict(xs, raw_score=True)) / 2 for b in boosters], 0)
        o = odds(p) + scale * d
        live = p[:, 0] + p[:, 1]
        h = live / (1 + np.exp(-o))
        out.append(np.stack([h, live - h, p[:, 2]], 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-brief", type=float, default=1.0)
    ap.add_argument("--with-base", nargs="?", const=True, default=False,
                    help="also read the base's probabilities; `--with-base only` leaves the reactions out (the control)")
    ap.add_argument("--target", default="both", choices=["both", "folded", "built"])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-sc", action="store_true", help="fit without SkillCorner's matches (its labels are the noisy ones)")
    a = ap.parse_args()
    ms = rdata.load_all()
    print("columns: spellF1 IDSSE Metrica SkillCorner mean (labels as built, then control-rule); brief = true control-rule possessions under 5 s")
    S.report("base (pw1e)", ms, [m.decode() for m in ms])
    targets = ["built", "folded"] if a.target == "both" else [a.target]
    per_seed = []
    for seed in a.seeds:
        bo = {prov: [fit([m for m in ms if m.fold != prov and not (a.no_sc and m.fold == "skillcorner")], t, seed, a.trees, a.weight_brief, a.with_base, a.lr) for t in targets] for prov in PROVS}
        per_seed.append(bo)
        for sc in a.scales:
            S.report(f"reactions x{sc:g} s{seed}", ms, [m.decode(apply(bo[m.fold], m, a.with_base, sc)) for m in ms])
    if len(a.seeds) > 1:
        for sc in a.scales:
            probs = {m.name: apply([b for bo in per_seed for b in bo[m.fold]], m, a.with_base, sc) for m in ms}
            S.report(f"reactions x{sc:g} avg", ms, [m.decode(probs[m.name]) for m in ms])
            if a.save and sc == 1.0:
                d = os.path.join(R.OUT, a.save)
                os.makedirs(d, exist_ok=True)
                for n, p in probs.items():
                    np.savez_compressed(os.path.join(d, n + ".npz"), **{f"p{i}": x.astype(np.float32) for i, x in enumerate(p)})


if __name__ == "__main__":
    main()
