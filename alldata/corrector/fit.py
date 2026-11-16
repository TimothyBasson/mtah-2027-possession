"""Train the corrector with one provider held out and save its probabilities as a run.

    python alldata/corrector/fit.py --tag c1 [--seed 0] [--feats feats_pw1e_v1] [--drop touch hold]

Gradient-boosted trees on cfeats rows, one model per label style (as built, folded). Writes
data/cache/alldata/runs/<tag>/<fold>/<match>.npz with keys s<i> / fs<i> (and the same under
p<i> / fp<i>), so alldata/learned/score.py and breakdown.py read it like any other run.
"""

import argparse
import glob
import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cfeats

CACHE = os.path.join(ROOT, "data", "cache", "alldata")
PROVS = ("idsse", "metrica", "skillcorner")
GROUPS = {"touch": lambda n: "_t" in n and n.split("_t")[-1][:1] in "pn" or n[:2] in ("tp", "tn"),
          "hold": lambda n: "hold" in n or "_vel_" in n or "margin" in n,
          "lags": lambda n: any(n.endswith(f"_{s}{l}") for s in "bf" for l in cfeats.LAGS),
          "run": lambda n: any(k in n for k in ("_run", "tostart", "toend", "_prev", "_next"))}


def load(sub, name):
    z = np.load(os.path.join(CACHE, "corrector", sub, name + ".npz"))
    names = [str(x) for x in z["names"]]
    nh = int(z["nhalves"])
    return [z[f"F_{i}"] for i in range(nh)], [z[f"y_{i}"] for i in range(nh)], [z[f"yf_{i}"] for i in range(nh)], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feats", default="feats_pw1e_v1")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--sc-share", type=float, default=0.25, help="SkillCorner's share of the training weight when it trains beside another provider")
    ap.add_argument("--folds", nargs="*", default=list(PROVS))
    a = ap.parse_args()
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE, "corrector", a.feats, "*.npz")))
    data = {n: load(a.feats, n) for n in names}
    cols = data[names[0]][3]
    keep = [i for i, c in enumerate(cols) if not any(GROUPS[g](c) for g in a.drop)]
    kcols = [cols[i] for i in keep]
    print(len(cols), "columns,", len(keep), "kept", flush=True)
    for fold in a.folds:
        tr = [n for n in names if not n.startswith(fold)]
        share = {p: 1.0 for p in PROVS if p != fold}
        if "skillcorner" in share:
            other = [p for p in share if p != "skillcorner"][0]
            share = {"skillcorner": a.sc_share, other: 1 - a.sc_share}
        tot = sum(share.values())
        rows = {p: sum(len(f) for n in tr if n.startswith(p) for f in data[n][0]) for p in share}
        X, Y, YF, Wt = [], [], [], []
        for n in tr:
            p = n.split("_")[0]
            for F, y, yf in zip(*data[n][:3]):
                for sw in (0, 1):
                    X.append((cfeats.swap(F, cols) if sw else F)[:, keep])
                    sy = (lambda v: np.where(v == 0, 1, np.where(v == 1, 0, v))) if sw else (lambda v: v)
                    Y.append(sy(y)); YF.append(sy(yf))
                    Wt.append(np.full(len(F), share[p] / tot / rows[p], np.float32))
        X, Y, YF, Wt = np.vstack(X), np.concatenate(Y), np.concatenate(YF), np.concatenate(Wt)
        Wt *= len(Wt) / Wt.sum()
        models = {}
        for key, y in (("s", Y), ("fs", YF)):
            ok = y >= 0
            ds = lgb.Dataset(X[ok], y[ok], weight=Wt[ok], feature_name=kcols, free_raw_data=False)
            models[key] = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=a.leaves, min_data_in_leaf=200,
                                         feature_fraction=0.6, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0, seed=a.seed,
                                         num_threads=4, verbose=-1), ds, a.rounds)
        out = os.path.join(CACHE, "runs", a.tag, fold)
        os.makedirs(out, exist_ok=True)
        for n in names:
            if not n.startswith(fold):
                continue
            res = {}
            for i, F in enumerate(data[n][0]):
                for key, m in models.items():
                    p1 = m.predict(F[:, keep])
                    p2 = m.predict(cfeats.swap(F, cols)[:, keep])[:, [1, 0, 2]]
                    res[f"{key}{i}"] = ((p1 + p2) / 2).astype(np.float32)
                res[f"p{i}"], res[f"fp{i}"] = res[f"s{i}"], res[f"fs{i}"]
            np.savez(os.path.join(out, n + ".npz"), kind="test", **res)
        if fold == a.folds[0]:
            for key, m in models.items():
                g = m.feature_importance("gain")
                o = np.argsort(-g)[:25]
                print(key, "top gain:", ", ".join(f"{kcols[j]} {g[j] / g.sum():.3f}" for j in o), flush=True)
        print("fold", fold, "done", flush=True)


if __name__ == "__main__":
    main()
