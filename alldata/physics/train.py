"""Boosted trees over the feature tables, held out by provider.

    python alldata/physics/train.py --run base [--mode barath] [--target raw] [--drop q_ t_]
    python alldata/physics/train.py --run base --fold none      # all 30, for BAR-ATH
    python alldata/physics/train.py --run base_cv --fold cv     # held out by match, all providers in training

For each provider, a classifier (H / A / D per 5 Hz row) is trained on the other two providers,
which count equally, and predicts the held-out provider's matches. Each match is read twice, as it
is and with the teams exchanged, in training and at prediction (the two readings are averaged), so
the classifier cannot favour "home". Probabilities go to data/cache/alldata/pred/<run>/.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from build import table_path  # noqa: E402

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_child_samples=200, colsample_bytree=0.5, subsample=0.7, subsample_freq=1,
              reg_lambda=1.0, n_estimators=400, n_jobs=4, random_state=0, deterministic=True,
              force_row_wise=True, verbose=-1)
SWAP_Y = np.array([1, 0, 2])


def pred_dir(run):
    return os.path.join(C.CACHE, "pred", run)


def keep_columns(names, drop):
    return np.array([not any(n.startswith(p) for p in drop) for n in names])


def load_tables(names, mode, tag, modes_extra=()):
    out = {}
    for n in names:
        z = np.load(table_path(n, mode, tag))
        out[n] = {k: z[k] for k in z.files}
    return out


def rows(tabs, names, target, stride, keep, extra_tabs=()):
    """Training rows of the named matches: both readings, graded rows only, provider-balanced."""
    X, y, w = [], [], []
    per = {}
    for src in (tabs,) + tuple(extra_tabs):
        for n in names:
            t = src[n]
            yy = t["yf" if target == "folded" else "y"].astype(int)
            ok = np.flatnonzero(yy >= 0)[::stride]
            X += [t["F"][ok][:, keep], t["Fs"][ok][:, keep]]
            y += [yy[ok], SWAP_Y[yy[ok]]]
            w += [np.full(2 * len(ok), 0.0)]
            per[C.provider(n)] = per.get(C.provider(n), 0) + 2 * len(ok)
            w[-1][:] = 1.0
            w[-1] = (C.provider(n), w[-1])
    wt = np.concatenate([v / per[p] for p, v in w])
    return np.concatenate(X), np.concatenate(y), wt * len(wt)


# What the dead-ball head may read. A dead ball in view exists in the outside matches only on
# SkillCorner; elsewhere it is made up (common.dead_ball_path), so how such a ball moves, who is
# near it and what touches it shows must not decide alive or dead. Where it is, whether it is
# shown, the players and the players-only prior may.
DEAD_COLUMNS = ("q_", "p_", "g_", "o_", "b_x", "b_absy", "b_side_margin", "b_end_margin")


class TwoHeads:
    """P(dead) from the restricted columns, P(home | alive) from all of them, on alive rows."""

    def __init__(self, names, params):
        import lightgbm as lgb
        self.dcols = np.array([n.startswith(DEAD_COLUMNS) and "speed" not in n.replace("p_speed", "")
                               and n != "g_lost_vout" for n in names])
        bp = dict(params, objective="binary")
        bp.pop("num_class")
        self.dead, self.team = lgb.LGBMClassifier(**bp), lgb.LGBMClassifier(**bp)

    def fit(self, X, y, w):
        self.dead.fit(X[:, self.dcols], y == 2, sample_weight=w)
        live = y < 2
        self.team.fit(X[live], y[live] == 0, sample_weight=w[live])
        return self

    def predict_proba(self, X):
        pd_ = self.dead.predict_proba(X[:, self.dcols])[:, 1]
        ph = self.team.predict_proba(X)[:, 1]
        return np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], 1)

    def importance(self):
        return self.team.booster_.feature_importance("gain")


def fit(X, y, w, params, names=None, heads=False):
    import lightgbm as lgb
    if heads:
        return TwoHeads(names, params).fit(X, y, w)
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y, sample_weight=w)
    return m


def predict(model, t, keep):
    p = model.predict_proba(t["F"][:, keep])
    if "Fs" in t:
        p = (p + model.predict_proba(t["Fs"][:, keep])[:, [1, 0, 2]]) / 2
    return p


def save_probs(run, name, t, p, kind="test"):
    os.makedirs(pred_dir(run), exist_ok=True)
    np.savez_compressed(os.path.join(pred_dir(run), name + ".npz"), kind=kind,
                        **{f"p{h}": p[t["half"] == h].astype(np.float32) for h in np.unique(t["half"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="barath")
    ap.add_argument("--tag", default="f1")
    ap.add_argument("--target", default="raw", choices=["raw", "folded"])
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fold", default="all")
    ap.add_argument("--trees", type=int, default=PARAMS["n_estimators"])
    ap.add_argument("--leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=PARAMS["n_jobs"])
    ap.add_argument("--heads", action="store_true", help="separate dead-ball and team heads (TwoHeads)")
    ap.add_argument("--test-mode", default=None, help="ball mode of the held-out matches (default: --mode)")
    a = ap.parse_args()
    params = dict(PARAMS, n_estimators=a.trees, num_leaves=a.leaves, random_state=a.seed, n_jobs=a.jobs)
    names = C.names()
    tabs = load_tables(names, a.mode, a.tag)
    test_tabs = tabs if not a.test_mode or a.test_mode == a.mode else load_tables(names, a.test_mode, a.tag)
    cols = tabs[names[0]]["names"]
    keep = keep_columns(cols, a.drop)
    print(f"{keep.sum()} of {len(cols)} features", flush=True)
    if a.fold == "none":
        import joblib
        X, y, w = rows(tabs, names, a.target, a.stride, keep)
        m = fit(X, y, w, params, cols[keep], a.heads)
        os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
        joblib.dump(dict(model=m, keep=keep, names=cols, args=vars(a)),
                    os.path.join(HERE, "models", a.run + ".joblib"), compress=3)
        z = np.load(table_path("bar_ath", a.mode, a.tag))
        t = {k: z[k] for k in z.files}
        save_probs(a.run, "bar_ath", t, predict(m, t, keep))
        return
    if a.fold == "cv":  # three folds of matches, every provider in training: what BAR-ATH's model sees
        order = {p: [n for n in names if C.provider(n) == p] for p in C.PROVIDERS}
        folds = [(f"cv{k}", [n for p in C.PROVIDERS for i, n in enumerate(order[p]) if i % 3 == k]) for k in range(3)]
    else:
        folds = [(p, [n for n in names if C.provider(n) == p]) for p in (C.PROVIDERS if a.fold == "all" else [a.fold])]
    for prov, held in folds:
        tr = [n for n in names if n not in held]
        X, y, w = rows(tabs, tr, a.target, a.stride, keep)
        m = fit(X, y, w, params, cols[keep], a.heads)
        for n in held:
            save_probs(a.run, n, test_tabs[n], predict(m, test_tabs[n], keep))
        imp = m.importance() if a.heads else m.booster_.feature_importance("gain")
        top = np.argsort(-imp)[:12]
        print(prov, len(y), "rows;", ", ".join(f"{cols[keep][i]} {imp[i] / imp.sum():.2f}" for i in top), flush=True)


if __name__ == "__main__":
    main()
