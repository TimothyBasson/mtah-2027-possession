"""Shared pieces for the post-classifier experiments: cached emissions (external/post_cache.py),
a compiled Viterbi, and the spell score against the control-folded labels."""

import glob
import os
import sys

import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import hmm.decode as dec  # noqa: E402
from hmm.model import rate_log_trans  # noqa: E402

POST = os.path.join(ROOT, "data", "post")
SETS = ("idsse", "metrica", "skillcorner")
H, A, D = 0, 1, 2


@njit(cache=True)
def viterbi(log_emit, log_trans, log_start):
    n, s = log_emit.shape
    back = np.empty((n, s), dtype=np.int8)
    score = log_start + log_emit[0]
    new = np.empty(s)
    for t in range(1, n):
        for j in range(s):
            best, arg = -1e300, 0
            for i in range(s):
                v = score[i] + log_trans[i, j]
                if v > best:
                    best, arg = v, i
            back[t, j] = arg
            new[j] = best + log_emit[t, j]
        for j in range(s):
            score[j] = new[j]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def load_all(sets=SETS):
    out = {}
    for p in sorted(glob.glob(os.path.join(POST, "*.npz"))):
        name = os.path.basename(p)[:-4]
        if name.startswith(tuple(sets)):
            m = dict(np.load(p))
            m["gt"] = pm.label_codes(m["folded"])
            m["gt_raw"] = pm.label_codes(m["labels"])
            m["runs"] = pm.valid_runs(m["gt"])
            out[name] = m
    return out


def transitions(fps, possession=dec.POSSESSION_SECONDS, stoppage=dec.STOPPAGE_SECONDS,
                to_dead=dec.TO_DEAD_SHARE):
    return rate_log_trans(possession * fps, stoppage * fps, to_dead)


def emissions(m, weight=dec.CONTROL_WEIGHT, ratio="ratio_control_loso"):
    return m["emit"].astype(np.float64) + weight * m[ratio].astype(np.float64)


def decode(m, emit=None, trans=None):
    """State path (0 H, 1 A, 2 D) on every frame, -1 outside the halves."""
    emit = emissions(m) if emit is None else emit
    trans = transitions(int(m["fps"])) if trans is None else trans
    path = np.full(len(emit), -1, dtype=np.int8)
    for lo, hi in m["halves"]:
        path[lo:hi] = viterbi(np.ascontiguousarray(emit[lo:hi]), trans, np.log(dec.START))
    return path


CODE = np.array([pm.H, pm.A, pm.D])


def to_codes(m, path):
    pred = np.where(path >= 0, CODE[np.clip(path, 0, 2)], pm.X)
    pred[m["gt_raw"] == pm.X] = pm.X
    return pred


def spell(m, path, gt="gt"):
    """spell_scores dict of a state path against the folded (or raw) labels."""
    pred = pm.fill_abstentions(to_codes(m, path), m["runs"])
    return pm.spell_scores(m[gt], pred, m["runs"])


def by_set(values):
    """{set: mean} and the mean of the three, from {match: value}."""
    out = {k: float(np.mean([v for n, v in values.items() if n.startswith(k)])) for k in SETS}
    out["mean"] = float(np.mean([out[k] for k in SETS]))
    return out


def fmt(r):
    return "{:.3f} / {:.3f} / {:.3f}  mean {:.4f}".format(*(r[k] for k in SETS), r["mean"])
