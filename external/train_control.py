"""Train the "who controls the ball" classifier (hmm/control.py) on the external matches.

    python external/control_features.py             # features, once (~3 min)
    python external/train_control.py                # -> hmm/models/control_state.npz
    python external/train_control.py --loso         # held-out posteriors for score_control.py

Needs scikit-learn, pinned in requirements-dev.txt. The decoder does not: the fitted trees are
exported to plain arrays (hmm/trees.py), the model is committed as hmm/models/control_state.npz
(it is derived from the open external data only), and the decoder evaluates it in numpy. Retrained
with scikit-learn 1.6.1 from the same features it comes out byte-identical; the sha256 printed at
the end is the check.

The target is the label per frame, H / A / D, folded towards the challenge's controlled-possession
rule (external/control.py): a spell under 2 s with fewer than 2 own touches goes back to the team
that had the ball. Trained on the labels as they are instead, the decoder scores the same against
the raw labels (held-out mean spellF1 gain +0.037, against +0.036) and worse against the folded
ones (+0.042, against +0.052), and the folded labels are the ones that follow the challenge's rule.
Every match counts in both team orientations (the
features computed with the teams swapped, the label swapped to match), so the classifier learns no
home or away bias. Rows are every STEP-th graded frame: neighbouring frames repeat each other.

--loso trains three models, each with one set (IDSSE, Metrica, SkillCorner) left out, and saves
the left-out set's posteriors, so external/score_control.py can score the decoder on matches the
classifier never saw. Without it, one model is trained on all 30 matches for BAR-ATH.

The model is a HistGradientBoostingClassifier with early stopping on a 10% validation split.
Settings were not swept: held-out log loss with every 5th frame is 0.28 / 0.53 / 1.10 (IDSSE /
Metrica / SkillCorner left out).
"""

import argparse
import hashlib
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
from control import control_labels  # noqa: E402
from control_features import CACHE  # noqa: E402
from hmm import trees  # noqa: E402

SETS = ("idsse", "metrica", "skillcorner")
STEP = 5
MODEL_PATH = os.path.join(ROOT, "hmm", "models", "control_state.npz")
SWAP = np.array([1, 0, 2])  # H <-> A, D stays


def code(labels):
    return np.select([labels == "H", labels == "A", labels == "D"], [0, 1, 2], -1)


def load_matches():
    matches = {}
    for path in sorted(glob.glob(os.path.join(CACHE, "control_*_*.npz"))):
        name = os.path.basename(path)[len("control_"):-4]
        if not name.startswith(SETS):
            continue
        z = np.load(path)
        labels = control_labels(dict(np.load(os.path.join(ROOT, "data", "external",
                                                          name + ".npz"))))
        matches[name] = (z["X"], z["Xs"], code(labels), list(z["names"]))
    if not matches:
        sys.exit("no features in {}: run external/control_features.py first".format(CACHE))
    return matches


def keep_columns(names, drop):
    """Indices of the feature columns to train on: all but those containing any of drop."""
    return [i for i, n in enumerate(names) if not any(d in n for d in drop)]


def rows(matches, names):
    X, y = [], []
    for n in names:
        a, b, labels, _ = matches[n]
        g = np.flatnonzero(labels >= 0)[::STEP]
        X += [a[g], b[g]]
        y += [labels[g], SWAP[labels[g]]]
    return np.vstack(X), np.concatenate(y)


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
        validation_fraction=0.1, random_state=0).fit(X, y)


def posteriors(model, a, b):
    return (model.predict_proba(a) + model.predict_proba(b)[:, SWAP]) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loso", action="store_true", help="held-out posteriors per set")
    ap.add_argument("--drop", default="", help="comma-separated substrings: leave out the feature "
                    "columns whose names contain any of them (an ablation)")
    ap.add_argument("--out", help="model path (default {}) or, with --loso, the posteriors' "
                    "file name in the cache".format(os.path.relpath(MODEL_PATH, ROOT)))
    a = ap.parse_args()
    matches = load_matches()
    names = next(iter(matches.values()))[3]
    keep = keep_columns(names, [d for d in a.drop.split(",") if d])
    if len(keep) < len(names):
        print("leaving out {} of {} columns".format(len(names) - len(keep), len(names)))
        matches = {n: (x[:, keep], xs[:, keep], y, nm) for n, (x, xs, y, nm) in matches.items()}
        names = [names[i] for i in keep]
    if a.loso:
        out = {}
        for k in SETS:
            train = [n for n in matches if not n.startswith(k)]
            X, y = rows(matches, train)
            t = time.time()
            model = fit(X, y)
            prior = np.bincount(y, minlength=3) / len(y)
            for n in (n for n in matches if n.startswith(k)):
                out[n] = posteriors(model, matches[n][0], matches[n][1]).astype(np.float32)
                out["prior_" + n] = prior
            print("{:12s} left out: {} rows, {} iterations, {:.0f} s".format(
                k, len(y), model.n_iter_, time.time() - t), flush=True)
        path = os.path.join(CACHE, a.out or "control_loso.npz")
        np.savez_compressed(path, **out)
        print("->", path)
        return
    X, y = rows(matches, list(matches))
    t = time.time()
    model = fit(X, y)
    print("{} rows, {} iterations, {:.0f} s".format(len(y), model.n_iter_, time.time() - t))
    path = a.out or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = np.bincount(y, minlength=3) / len(y)
    trees.export(model, path, prior=prior, names=np.array(names))
    check = X[np.random.default_rng(0).choice(len(X), 20000, replace=False)]
    diff = np.abs(trees.predict_proba(trees.load(path), check) - model.predict_proba(check))
    print("numpy trees vs scikit-learn: max |p difference| {:.2e} on 20000 rows".format(diff.max()))
    if diff.max() > 1e-6:
        sys.exit("the exported trees do not reproduce the model")
    with open(path, "rb") as handle:
        print("->", path, "sha256", hashlib.sha256(handle.read()).hexdigest())


if __name__ == "__main__":
    main()
