"""Shared pieces for the turnover work: loading held-out probabilities, a decode whose H<->A
switching cost can vary by frame, and scoring on the labels as given and folded."""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402

LETTERS = np.array(list("HAD"))
PROVIDERS = ("idsse", "metrica", "skillcorner")
COST, MIN_TEAM, MIN_DEAD = 4, 1, 1  # the shipped decode (deep/run.py)


def paths():
    return sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))


def base_probs(name, tags, n_halves):
    """Held-out probabilities of a match, averaged over tags (seeds): [(T, 3)] per half at 5 fps."""
    ps = [np.load(os.path.join(ROOT, "data", "cache", "deep", t, name.split("_")[0], name + ".npz")) for t in tags]
    return [np.mean([p[f"p{i}"] for p in ps], 0) for i in range(n_halves)]


def viterbi_ev(logp, cost, bonus=None, ha_cost=None):
    """Best path through (n, 3) log-probabilities. Any change of state costs `cost` (a direct
    H<->A change `ha_cost` when given), less the turnover evidence at that frame: bonus is (n,)
    for either direction or (n, 2) for [to home, to away]."""
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    v = logp[0].copy()
    base = -cost * (1 - np.eye(3))
    if ha_cost is not None:
        base[0, 1] = base[1, 0] = -ha_cost
    if bonus is not None and bonus.ndim == 1:
        bonus = np.stack([bonus, bonus], 1)
    hot = np.zeros(n, bool) if bonus is None else (bonus > 0).any(1)
    for i in range(1, n):
        tr = base
        if hot[i]:
            tr = base.copy()
            tr[1, 0] += bonus[i, 0]   # away -> home
            tr[0, 1] += bonus[i, 1]   # home -> away
        s = v[:, None] + tr          # from, to
        back[i] = s.argmax(0)
        v = s.max(0) + logp[i]
    path = np.zeros(n, np.int8)
    path[-1] = v.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def labels_ev(probs, spans, n, bonus=None, cost=COST, min_team=MIN_TEAM, min_dead=MIN_DEAD, ha_cost=None):
    """decode.labels_from with turnover evidence: bonus is a per-frame (25 fps, whole match) array."""
    out = np.full(n, "X", "<U1")
    for p, (a, b) in zip(probs, spans):
        lp = np.log(np.clip(decode.upsample(p, b - a + 1), 1e-6, 1))
        path = viterbi_ev(lp, cost, None if bonus is None else bonus[a:b + 1], ha_cost)
        if min_team or min_dead:
            path = decode.absorb(path, {0: min_team * 25, 1: min_team * 25, 2: min_dead * 25})
        out[a:b + 1] = LETTERS[path]
    return out


def turnovers(labels):
    """Frames i where labels[i-1], labels[i] are H, A or A, H."""
    l = np.asarray(labels)
    return np.flatnonzero(((l[1:] == "H") & (l[:-1] == "A")) | ((l[1:] == "A") & (l[:-1] == "H"))) + 1


def provider_mean(names, vals):
    vals = np.asarray(vals, float)
    per = [vals[[n.startswith(p) for n in names]].mean(0) for p in PROVIDERS]
    return np.array(per + [np.mean(per, 0)])
