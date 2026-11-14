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
from hmm.control import hold  # noqa: E402

SETS = ("idsse", "metrica", "skillcorner")
ITERATIONS = 300
MODEL_PATH = os.path.join(ROOT, "hmm", "models", "control_state.npz")
SWAP = np.array([1, 0, 2])  # H <-> A, D stays


def code(labels):
    return np.select([labels == "H", labels == "A", labels == "D"], [0, 1, 2], -1)


def load_matches():
    """{match: (X, Xs, label codes on the cached frames, column names, frames, frames in the match)}"""
    matches = {}
    for path in sorted(glob.glob(os.path.join(CACHE, "control_v8_*_*.npz"))):
        name = os.path.basename(path)[len("control_v8_"):-4]
        if not name.startswith(SETS):
            continue
        z = np.load(path)
        labels = control_labels(dict(np.load(os.path.join(ROOT, "data", "external",
                                                          name + ".npz"))))
        matches[name] = (z["X"], z["Xs"], code(labels)[z["frames"]], list(z["names"]),
                         z["frames"], len(labels))
    if not matches:
        sys.exit("no features in {}: run external/control_features.py first".format(CACHE))
    return matches


def keep_columns(names, drop):
    """Indices of the feature columns to train on: all but those containing any of drop."""
    return [i for i, n in enumerate(names) if not any(d in n for d in drop)]


def rows(matches, names):
    """(X, y, weight): every graded cached frame of these matches, in both team orientations. The
    weights make each provider count the same, however many matches it has: unweighted, IDSSE and
    SkillCorner would outvote Metrica's three."""
    X, y, w = [], [], []
    graded = {k: sum(int(np.sum(matches[n][2] >= 0)) for n in names if n.startswith(k))
              for k in SETS}
    for n in names:
        a, b, labels = matches[n][:3]
        g = np.flatnonzero(labels >= 0)
        X += [a[g], b[g]]
        y += [labels[g], SWAP[labels[g]]]
        w.append(np.full(2 * len(g), 1.0 / graded[n.split("_")[0]]))
    w = np.concatenate(w)
    return np.vstack(X), np.concatenate(y), w / w.mean()


def fit(X, y, weight):
    """No early stopping and a fixed seed, so a refit on the same rows gives the same trees. The
    leaf floor and the L2 penalty are seq/model.py's, taken as they were and not swept here."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=ITERATIONS, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0).fit(X, y, sample_weight=weight)


def class_shares(y, weight):
    return np.bincount(y, weights=weight, minlength=3) / weight.sum()


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
        matches = {n: (m[0][:, keep], m[1][:, keep]) + m[2:] for n, m in matches.items()}
        names = [names[i] for i in keep]
    if a.loso:
        out = {}
        for k in SETS:
            train = [n for n in matches if not n.startswith(k)]
            X, y, w = rows(matches, train)
            t = time.time()
            model = fit(X, y, w)
            prior = class_shares(y, w)
            loss = []
            for n in (n for n in matches if n.startswith(k)):
                a, b, labels, _, frames, n_frames = matches[n]
                p = posteriors(model, a, b).astype(np.float32)
                g = labels >= 0
                loss.append(-np.mean(np.log(np.clip(p[g, labels[g]], 1e-6, 1))))
                out[n] = hold(p, n_frames)
                out["prior_" + n] = prior
            print("{:12s} left out: {} rows, {:.0f} s, held-out log loss {:.2f}".format(
                k, len(y), time.time() - t, np.mean(loss)), flush=True)
        path = os.path.join(CACHE, a.out or "control_loso.npz")
        np.savez_compressed(path, **out)
        print("->", path)
        return
    X, y, w = rows(matches, list(matches))
    t = time.time()
    model = fit(X, y, w)
    print("{} rows, {:.0f} s".format(len(y), time.time() - t))
    path = a.out or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = class_shares(y, w)
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
