"""LOSO joint trainer. usage: train.py OUT [--drop a,b] [--keep prefixes] [--bal] [--reg] [--raw] [--iters N]"""
from lib import *
import argparse, time
J = ROOT + "/data/cache/wip/joint"
SWAP = np.array([1, 0, 2])
def code(l): return np.select([l == "H", l == "A", l == "D"], [0, 1, 2], -1)
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--drop", default=""); ap.add_argument("--keep", default="")
ap.add_argument("--bal", action="store_true"); ap.add_argument("--reg", action="store_true"); ap.add_argument("--raw", action="store_true")
ap.add_argument("--iters", type=int, default=300); ap.add_argument("--leaf", type=int, default=200); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--sets", default="idsse,metrica,skillcorner")
a = ap.parse_args()
if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    Z = {n: np.load(f"{J}/{n}.npz") for n in NAMES}
    names = list(Z[NAMES[0]]["names"])
    drop = [d for d in a.drop.split(",") if d]; keep = [k for k in a.keep.split(",") if k]
    cols = [i for i, n in enumerate(names) if not any(d in n for d in drop) and (not keep or any(n.startswith(k) for k in keep))]
    print(len(cols), "of", len(names), "columns", flush=True)
    D = {}
    for n in NAMES:
        z = Z[n]; D[n] = (z["X"][:, cols], z["Xs"][:, cols], code(z["yraw"] if a.raw else z["y"]), z["frames"])
    out = {}
    for k in a.sets.split(","):
        tr = [n for n in NAMES if not n.startswith(k)]
        X, y, w = [], [], []
        nprov = {s: sum(len(D[n][2]) for n in tr if n.startswith(s)) for s in SETS if s != k}
        for n in tr:
            A, B, yy, _ = D[n]; g = yy >= 0
            X += [A[g], B[g]]; y += [yy[g], SWAP[yy[g]]]
            ww = 1.0 / nprov[n.split("_")[0]] if a.bal else 1.0
            w += [np.full(2 * g.sum(), ww)]
        X = np.vstack(X); y = np.concatenate(y); w = np.concatenate(w); w = w / w.mean()
        t = time.time()
        kw = dict(min_samples_leaf=a.leaf, l2_regularization=1.0) if a.reg else {}
        clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=63, early_stopping=False, random_state=0, **kw).fit(X, y, sample_weight=w)
        prior = np.bincount(y, weights=w, minlength=3) / w.sum()
        ll = []
        for n in (n for n in NAMES if n.startswith(k)):
            A, B, yy, fr = D[n]
            p = ((clf.predict_proba(A) + clf.predict_proba(B)[:, SWAP]) / 2).astype(np.float32)
            g = yy >= 0; ll.append(-np.mean(np.log(np.clip(p[g, yy[g]], 1e-6, 1))))
            nfr = len(load(n)["labels"])
            out[n] = p[np.clip((np.arange(nfr) + 2) // 5, 0, len(fr) - 1)]; out["prior_" + n] = prior
        print(f"{k} left out: {len(y)} rows {time.time()-t:.0f}s  held-out logloss {np.mean(ll):.3f}", flush=True)
    np.savez_compressed(f"{ROOT}/data/cache/{a.out}", **out)
