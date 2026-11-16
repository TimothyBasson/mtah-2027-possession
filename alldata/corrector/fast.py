"""The base decode, compiled: the same three-state Viterbi and short-run absorption as
deep/decode.py (copied from turnover/common.py, which checked they give identical labels)."""

import numpy as np


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



try:
    import numba
    _viterbi = numba.njit(cache=True)(_viterbi)
    _absorb = numba.njit(cache=True)(_absorb)
except ImportError:
    pass


def decode_path(logp, cost, min_team, min_dead, fps=25):
    base = -float(cost) * (1 - np.eye(3))
    path = _viterbi(np.ascontiguousarray(logp, dtype=np.float64), base, np.zeros((len(logp), 2)))
    return _absorb(np.asarray(path, np.int8), np.array([min_team * fps, min_team * fps, min_dead * fps], np.int64))
