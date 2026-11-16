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


def _viterbi(logp, base, bonus):
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    v = logp[0].copy()
    for i in range(1, n):
        nv = np.empty(3)
        for to in range(3):
            best, arg = -1e18, 0
            for fr in range(3):
                s = v[fr] + base[fr, to]
                if fr == 1 and to == 0:
                    s += bonus[i, 0]   # away -> home
                elif fr == 0 and to == 1:
                    s += bonus[i, 1]   # home -> away
                if s > best:
                    best, arg = s, fr
            back[i, to] = arg
            nv[to] = best + logp[i, to]
        v = nv
    path = np.zeros(n, np.int8)
    path[-1] = np.argmax(v)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _absorb(codes, min_len):
    """decode.absorb, the same result: runs shorter than min_len[state] join the longer neighbour,
    shortest first (the first of equals), then equal neighbours merge."""
    n = len(codes)
    k = np.empty(n, np.int8); m = np.empty(n, np.int64)
    r = 0
    for i in range(n):
        if r > 0 and k[r - 1] == codes[i]:
            m[r - 1] += 1
        else:
            k[r] = codes[i]; m[r] = 1; r += 1
    while r > 1:
        best = -1
        for i in range(r):
            if m[i] < min_len[k[i]] and (best < 0 or m[i] < m[best]):
                best = i
        if best < 0:
            break
        if best == 0:
            j = 1
        elif best == r - 1:
            j = best - 1
        else:
            j = best - 1 if m[best - 1] >= m[best + 1] else best + 1
        m[j] += m[best]
        for i in range(best, r - 1):
            k[i] = k[i + 1]; m[i] = m[i + 1]
        r -= 1
        w = 0
        for i in range(1, r):
            if k[i] == k[w]:
                m[w] += m[i]
            else:
                w += 1; k[w] = k[i]; m[w] = m[i]
        r = w + 1
    out = np.empty(n, np.int8)
    pos = 0
    for i in range(r):
        out[pos:pos + m[i]] = k[i]
        pos += m[i]
    return out


def absorb(codes, min_team, min_dead):
    return _absorb(np.asarray(codes, np.int8), np.array([min_team * 25, min_team * 25, min_dead * 25], np.int64))


try:  # the loop is 100 times faster compiled; the result is the same
    import numba
    _viterbi = numba.njit(cache=True)(_viterbi)
    _absorb = numba.njit(cache=True)(_absorb)
except ImportError:
    pass


def viterbi_ev(logp, cost, bonus=None, ha_cost=None):
    """Best path through (n, 3) log-probabilities. Any change of state costs `cost` (a direct
    H<->A change `ha_cost` when given), less the turnover evidence at that frame: bonus is (n,)
    for either direction or (n, 2) for [to home, to away]."""
    base = -float(cost) * (1 - np.eye(3))
    if ha_cost is not None:
        base[0, 1] = base[1, 0] = -float(ha_cost)
    if bonus is None:
        bonus = np.zeros((len(logp), 2))
    elif bonus.ndim == 1:
        bonus = np.stack([bonus, bonus], 1)
    return _viterbi(np.ascontiguousarray(logp, dtype=np.float64), base, np.ascontiguousarray(bonus, dtype=np.float64))


def labels_ev(probs, spans, n, bonus=None, cost=COST, min_team=MIN_TEAM, min_dead=MIN_DEAD, ha_cost=None):
    """decode.labels_from with turnover evidence: bonus is a per-frame (25 fps, whole match) array."""
    out = np.full(n, "X", "<U1")
    for p, (a, b) in zip(probs, spans):
        lp = np.log(np.clip(decode.upsample(p, b - a + 1), 1e-6, 1))
        path = viterbi_ev(lp, cost, None if bonus is None else bonus[a:b + 1], ha_cost)
        if min_team or min_dead:
            path = absorb(path, min_team, min_dead)
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
