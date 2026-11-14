"""Gradient-boosted trees evaluated in numpy, so the decoder needs no scikit-learn.

external/train_control.py fits a scikit-learn HistGradientBoostingClassifier and saves it with
export(); load() and predict_proba() here read it back and reproduce the classifier's
predict_proba to float precision (checked when the model is exported).

The saved arrays, one row per node over all trees: feature, threshold, missing_left, left, right
(indices within the tree), is_leaf, value. tree_start gives each tree's first node, tree_class the
class its raw score adds to; baseline is the per-class starting score.
"""

import numpy as np

FIELDS = ("feature", "threshold", "missing_left", "left", "right", "is_leaf", "value")


def export(model, path, **extra):
    """Save a fitted HistGradientBoostingClassifier (numeric features only) as plain arrays."""
    rows = {k: [] for k in FIELDS}
    tree_start, tree_class = [], []
    offset = 0
    for per_class in model._predictors:
        for k, predictor in enumerate(per_class):
            nodes = predictor.nodes
            if np.any(nodes["is_categorical"]):
                raise ValueError("categorical splits are not supported")
            tree_start.append(offset)
            tree_class.append(k)
            rows["feature"].append(nodes["feature_idx"].astype(np.int32))
            rows["threshold"].append(nodes["num_threshold"].astype(np.float64))
            rows["missing_left"].append(nodes["missing_go_to_left"].astype(bool))
            rows["left"].append(nodes["left"].astype(np.int32))
            rows["right"].append(nodes["right"].astype(np.int32))
            rows["is_leaf"].append(nodes["is_leaf"].astype(bool))
            rows["value"].append(nodes["value"].astype(np.float64))
            offset += len(nodes)
    np.savez_compressed(path, **{k: np.concatenate(v) for k, v in rows.items()},
                        tree_start=np.array(tree_start, np.int64),
                        tree_class=np.array(tree_class, np.int32),
                        baseline=np.asarray(model._baseline_prediction, np.float64).ravel(),
                        classes=np.asarray(model.classes_), **extra)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def raw_scores(model, X):
    """(samples, classes) summed tree outputs plus the baseline."""
    X = np.asarray(X, np.float64)
    n_classes = len(model["classes"])
    out = np.tile(model["baseline"][:n_classes], (len(X), 1)) if n_classes > 1 else \
        np.full((len(X), 1), model["baseline"][0])
    starts = np.r_[model["tree_start"], len(model["feature"])]
    rows = np.arange(len(X))
    for t, k in enumerate(model["tree_class"]):
        lo, hi = starts[t], starts[t + 1]
        feature, threshold = model["feature"][lo:hi], model["threshold"][lo:hi]
        missing_left, leaf = model["missing_left"][lo:hi], model["is_leaf"][lo:hi]
        left, right, value = model["left"][lo:hi], model["right"][lo:hi], model["value"][lo:hi]
        node = np.zeros(len(X), np.int64)
        active = ~leaf[node]
        while np.any(active):
            idx = rows[active]
            nd = node[idx]
            x = X[idx, feature[nd]]
            go_left = np.where(np.isnan(x), missing_left[nd], x <= threshold[nd])
            node[idx] = np.where(go_left, left[nd], right[nd])
            active[idx] = ~leaf[node[idx]]
        out[:, k] += value[node]
    return out


def predict_proba(model, X):
    """Class probabilities, as HistGradientBoostingClassifier.predict_proba."""
    raw = raw_scores(model, X)
    if raw.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-raw[:, 0]))
        return np.column_stack([1 - p, p])
    raw -= raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(axis=1, keepdims=True)
