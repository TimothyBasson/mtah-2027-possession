"""Shared by the heads in this folder: the frozen features of a fold, both label views, scoring.

Labels: "raw" as the sources give them, "control" folded towards the challenge's
controlled-possession rule (external/control.py). Everything is scored with the organisers'
evaluation/possession_metrics.py at 25 fps.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEEP = os.path.dirname(HERE)
ROOT = os.path.dirname(DEEP)
sys.path[:0] = [DEEP, os.path.join(ROOT, "evaluation")]
import data as D  # noqa: E402
import possession_metrics as pm  # noqa: E402
from decode import absorb  # noqa: E402
from train import split  # noqa: E402

FOLDS = ("idsse", "metrica", "skillcorner")
LETTERS = np.array(list("HAD"))
VIEWS = ("00", "01", "10", "11")  # team swap, mirror
SWAP = np.array([1, 0, 2])


def labels_all():
    """Per match: y (raw) and yc (control) at 5 fps per half, spans, and both 25 fps label arrays."""
    import pickle
    cache = os.path.join(ROOT, "data", "cache", "spell", "labels.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    out = []
    for m, c in zip(D.load_all(), D.load_all(control=True)):
        out.append(dict(name=m["name"], provider=m["provider"], spans=m["spans"],
                        halves=[dict(y=h["y"], yc=hc["y"]) for h, hc in zip(m["halves"], c["halves"])],
                        labels=dict(raw=m["labels"], control=c["labels"])))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return out


def load_fold(fold, tag="v2", hidden=True):
    """(train, val, test): matches with, per half, y (raw) and yc (control) at 5 fps, the net's
    logits and (hidden) its 128-d state under the four symmetric views."""
    out = labels_all()
    for m in out:
        z = np.load(os.path.join(ROOT, "data", "cache", "spell", tag, fold, m["name"] + ".npz"))
        for i, e in enumerate(m["halves"]):
            for v in VIEWS:
                e["l" + v] = z[f"l{i}_{v}"]
                if hidden:
                    e["h" + v] = z[f"h{i}_{v}"]
    return split(out, fold)


def tta(per_view):
    """Average class scores over the four views; a swapped view names the teams the other way."""
    return np.mean([per_view[v][:, SWAP] if v[0] == "1" else per_view[v] for v in VIEWS], 0)


def to25(path5, n):
    """5 fps states -> n frames at 25 fps, each frame taking the nearest step."""
    return path5[np.minimum((np.arange(n) + D.STEP // 2) // D.STEP, len(path5) - 1)]


def spell_f1(match, paths5, view):
    """spellF1 of one match from its per-half 5 fps state paths, against the raw or control labels."""
    truth = match["labels"][view]
    pred = np.full(len(truth), "X", "<U1")
    for p, (a, b) in zip(paths5, match["spans"]):
        pred[a:b + 1] = LETTERS[to25(np.asarray(p), b - a + 1)]
    gt, pr = pm.label_codes(truth), pm.label_codes(pred)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pr, runs), runs)["mean_f1"]


def report(name, rows):
    """rows: {fold: {view: score}} -> one line, the three providers and their mean for each view."""
    line = f"{name:44s}"
    for view in ("raw", "control"):
        s = [rows[f][view] for f in FOLDS if f in rows]
        line += f" {view[:3]} " + " ".join(f"{x:.3f}" for x in s) + f" | {np.mean(s):.3f}  "
    print(line, flush=True)
