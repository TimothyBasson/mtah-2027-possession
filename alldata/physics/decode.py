"""Per-frame probabilities (5 Hz) -> one label per frame.

A three-state chain (H, A, D) with a cost for any change of state, Viterbi over the half on log
probabilities raised to 25 fps, then runs shorter than a minimum joined to the longer neighbour.
"""

import itertools

import numpy as np
from numba import njit

from common import STEP

LETTERS = np.array(list("HAD"))


def upsample(p, n, step=STEP):
    t = np.arange(len(p)) * step
    return np.stack([np.interp(np.arange(n), t, p[:, k]) for k in range(p.shape[1])], 1)


@njit(cache=True)
def viterbi(logp, cost):
    """Best path under a cost matrix cost[i, j] for moving from state i to j."""
    n, S = logp.shape
    back = np.zeros((n, S), np.int8)
    v = logp[0].copy()
    nv = np.zeros(S)
    for i in range(1, n):
        for j in range(S):
            best, arg = v[j], j
            for k in range(S):
                c = v[k] - cost[k, j]
                if c > best:
                    best, arg = c, k
            nv[j] = best + logp[i, j]
            back[i, j] = arg
        for j in range(S):
            v[j] = nv[j]
    path = np.zeros(n, np.int8)
    path[n - 1] = np.argmax(v)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def absorb(codes, min_len):
    """Runs shorter than min_len[state] frames join the longer neighbour, shortest first."""
    runs = [[k, len(list(g))] for k, g in itertools.groupby(codes)]
    while len(runs) > 1:
        short = [i for i, (k, n) in enumerate(runs) if n < min_len[k]]
        if not short:
            break
        i = min(short, key=lambda i: runs[i][1])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: runs[j][1])
        runs[j][1] += runs.pop(i)[1]
        k = 1
        while k < len(runs):
            if runs[k - 1][0] == runs[k][0]:
                runs[k - 1][1] += runs.pop(k)[1]
            else:
                k += 1
    return np.array([k for k, n in runs for _ in range(n)], np.int8)


def cost_matrix(cost, cost_dead=None):
    """cost for a change of team, cost_dead for a change into or out of dead ball."""
    cd = cost if cost_dead is None else cost_dead
    return np.array([[0, cost, cd], [cost, 0, cd], [cd, cd, 0]], float)


def decode_half(p, n, cost=4.0, cost_dead=None, min_team=1.0, min_dead=1.0, power=1.0, fps=25):
    lp = power * np.log(np.clip(upsample(p, n), 1e-6, 1))
    path = viterbi(lp, cost_matrix(cost, cost_dead)) if cost > 0 else lp.argmax(1).astype(np.int8)
    if min_team or min_dead:
        path = absorb(path, {0: min_team * fps, 1: min_team * fps, 2: min_dead * fps})
    return path


def labels_from(probs, spans, n, **kw):
    out = np.full(n, "X", "<U1")
    for p, (a, b) in zip(probs, spans):
        out[a:b + 1] = LETTERS[decode_half(p, b - a + 1, **kw)]
    return out
