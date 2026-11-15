"""Held-out posteriors with adjustable classifier settings; saves one loso file per iteration checkpoint."""
import argparse, os, sys, time, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "external")]
import train_control as tc
from control import control_labels
from control_features import CACHE
from hmm.control import hold

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--iters", default="300")
ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--min-leaf", type=int, default=200)
ap.add_argument("--l2", type=float, default=1.0)
ap.add_argument("--max-features", type=float, default=1.0)
ap.add_argument("--max-depth", type=int, default=0)
ap.add_argument("--fold-seconds", type=float, default=2.0)
ap.add_argument("--raw-target", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--drop", default="")
ap.add_argument("--sets", default="idsse,metrica,skillcorner")
ap.add_argument("--prov-weights", default="", help="e.g. idsse=1,metrica=1,skillcorner=2")
ap.add_argument("--norm", default="", help="substrings of columns to divide by their per-match median")
ap.add_argument("--indomain", action="store_true")
a = ap.parse_args()
iters = sorted(int(i) for i in a.iters.split(","))

def load():
    m = {}
    for path in sorted(glob.glob(os.path.join(CACHE, "control_v8_*_*.npz"))):
        name = os.path.basename(path)[len("control_v8_"):-4]
        z = np.load(path)
        d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
        labels = d["labels"] if a.raw_target else control_labels(d, a.fold_seconds)
        X, Xs = z["X"], z["Xs"]
        if a.norm:
            cols = [i for i, n in enumerate(z["names"]) if any(s in n for s in a.norm.split(","))]
            for A in (X, Xs):
                med = np.nanmedian(A[:, cols], axis=0)
                A[:, cols] = A[:, cols] / np.where(med > 0, med, 1)
        m[name] = (X, Xs, tc.code(labels)[z["frames"]], list(z["names"]), z["frames"], len(labels))
    return m

matches = load()
names = next(iter(matches.values()))[3]
keep = tc.keep_columns(names, [d for d in a.drop.split(",") if d])
if len(keep) < len(names):
    print("leaving out", len(names) - len(keep), "columns")
    matches = {n: (m[0][:, keep], m[1][:, keep]) + m[2:] for n, m in matches.items()}
pw = dict(kv.split("=") for kv in a.prov_weights.split(",") if kv)
from sklearn.ensemble import HistGradientBoostingClassifier
outs = {i: {} for i in iters}
jobs = []
for k in a.sets.split(","):
    own = [n for n in matches if n.startswith(k)]
    if a.indomain:
        jobs += [(k, own[1::2], own[0::2]), (k, own[0::2], own[1::2])]
    else:
        jobs.append((k, [], own))
for k, extra, test in jobs:
    train = [n for n in matches if not n.startswith(k)] + extra
    X, y, w = tc.rows(matches, train)
    if pw:
        f = np.concatenate([np.full(2 * int(np.sum(matches[n][2] >= 0)), float(pw.get(n.split("_")[0], 1))) for n in train])
        w = w * f; w = w / w.mean()
    t = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=iters[-1], learning_rate=a.lr, max_leaf_nodes=a.leaves, min_samples_leaf=a.min_leaf,
        l2_regularization=a.l2, max_features=a.max_features, max_depth=a.max_depth or None,
        early_stopping=False, random_state=a.seed).fit(X, y, sample_weight=w)
    prior = tc.class_shares(y, w)
    del X
    loss = {i: [] for i in iters}
    for n in test:
        x, xs, labels, _, frames, n_frames = matches[n]
        g = labels >= 0
        for it, (p, q) in enumerate(zip(model.staged_predict_proba(x), model.staged_predict_proba(xs)), 1):
            if it in outs:
                p = ((p + q[:, tc.SWAP]) / 2).astype(np.float32)
                loss[it].append(-np.mean(np.log(np.clip(p[g, labels[g]], 1e-6, 1))))
                outs[it][n] = hold(p, n_frames)
                outs[it]["prior_" + n] = prior
    print(k, "left out: %d rows, %.0f s, log loss" % (len(y), time.time() - t),
          {i: round(float(np.mean(v)), 3) for i, v in loss.items()}, flush=True)
for i in iters:
    np.savez(os.path.join(CACHE, "loso_%s_%d.npz" % (a.tag, i)), **outs[i])
print("done", a.tag)
