"""Shared pieces for the team-naming step: matches, folded truth, stage-1 probabilities, scoring."""

import glob
import itertools
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

PROVIDERS = ("idsse", "metrica", "skillcorner")
DECODE = dict(cost=4, min_team=2, min_dead=3)  # the folded-label decode (evidence review)


def paths():
    return sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))


def truth(name):
    """(raw labels, labels folded to the controlled-possession rule, halves), cached."""
    cache = os.path.join(ROOT, "data", "cache", "team", "truth", name + ".pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    from control import control_labels
    d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
    out = (d["labels"], control_labels(d), [tuple(int(v) for v in h) for h in d["halves"]])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def stage1(tag, name, fold=None):
    """Stage-1 H/A/D probabilities at 5 fps per half, from the fold net that never saw the match's
    provider (or from the named fold)."""
    p = np.load(os.path.join(ROOT, "data", "cache", "deep", tag, fold or name.split("_")[0], name + ".npz"))
    return [p[k] for k in sorted(k for k in p.files if k[0] == "p")]


def spell_f1(gt, pred):
    pred = pred.copy()
    pred[gt == "X"] = "X"
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)["spell_f1_mean"]


def runs(codes):
    """[(start, stop, value)] of equal runs."""
    edge = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    s = np.r_[0, edge]
    e = np.r_[edge, len(codes)]
    return [(int(a), int(b), codes[a]) for a, b in zip(s, e)]


def provider_mean(names, vals):
    vals = np.asarray(vals, float)
    per = [vals[[n.startswith(p) for n in names]].mean(0) for p in PROVIDERS]
    return per + [np.mean(per, 0)]
