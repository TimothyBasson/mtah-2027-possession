"""Train the keeper-end detector: p(defending team has it, other team has it, dead | end features)
on gated rows. --loso: one fit per provider on the other two, held-out predictions saved for every
match; --all: one fit on all 30 for BAR-ATH. -> data/cache/keeper_pred.npz, hmm/models/keeper.pkl"""
import argparse, glob, os, pickle, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache", "keeper")
SETS = ("idsse", "metrica", "skillcorner")


def load():
    out = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        out[os.path.basename(p)[:-4]] = dict(np.load(p))
    return out


def rows(data, names, ball):
    X, y, w = [], [], []
    per = {k: sum(1 for n in names if n.startswith(k)) for k in SETS}
    for n in names:
        z = data[n]
        g = z[ball + "_gate"] & (z["y"] >= 0)
        X.append(z[ball + "_X"][g]); y.append(z["y"][g])
        k = n.split("_")[0]
        w.append(np.full(g.sum(), 1.0 / per[k] / max(g.sum(), 1)))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def fit(X, y, w, args):
    m = HistGradientBoostingClassifier(max_iter=args.rounds, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    m.fit(X, y, sample_weight=w / w.mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ball", default="hands", help="hands, score or both")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to drop")
    ap.add_argument("--out", default="keeper_pred")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    data = load()
    names_all = list(data)
    fnames = list(next(iter(data.values()))["names"])
    keep = [i for i, f in enumerate(fnames) if not any(f.startswith(d) for d in a.drop.split(",") if d)]
    balls = ("hands", "score") if a.train_ball == "both" else (a.train_ball,)
    if a.all:
        X, y, w = (np.concatenate(v) for v in zip(*[rows(data, names_all, b) for b in balls]))
        m = fit(X[:, keep], y, w, a)
        prior = np.array([np.sum(w * (y == c)) for c in range(3)]) / w.sum()
        with open(os.path.join(ROOT, "hmm", "models", "keeper.pkl"), "wb") as f:
            pickle.dump(dict(model=m, names=[fnames[i] for i in keep], prior=prior), f)
        print("all-match fit, prior", prior.round(3))
        return
    pred = {}
    for k in SETS:
        tr = [n for n in names_all if not n.startswith(k)]
        te = [n for n in names_all if n.startswith(k)]
        X, y, w = (np.concatenate(v) for v in zip(*[rows(data, tr, b) for b in balls]))
        m = fit(X[:, keep], y, w, a)
        prior = np.array([np.sum(w * (y == c)) for c in range(3)]) / w.sum()
        for ball in ("score", "hands"):
            Xt, yt, wt = rows(data, te, ball)
            p = m.predict_proba(Xt[:, keep])
            print("{:12s} {:5s} rows {:6d} share D/def/att {} logloss {:.3f} vs prior {:.3f} acc {:.3f}".format(
                k, ball, len(yt), np.bincount(yt, minlength=3)[[2, 0, 1]] / len(yt),
                log_loss(yt, p, labels=[0, 1, 2]),
                log_loss(yt, np.tile(prior, (len(yt), 1)), labels=[0, 1, 2]),
                np.mean(p.argmax(1) == yt)), flush=True)
        for n in te:
            z = data[n]
            for ball in ("score", "hands"):
                pred["{}|{}|p".format(n, ball)] = m.predict_proba(z[ball + "_X"][:, keep]).astype(np.float32)
            pred[n + "|prior"] = prior
    np.savez(os.path.join(ROOT, "data", "cache", a.out + ".npz"), **pred)


if __name__ == "__main__":
    main()
