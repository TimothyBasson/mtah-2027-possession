"""Decodes for the all-data model's frame probabilities that can keep a brief possession.

decode_half(p5, n, **params) -> codes (n,) at 25 fps. With the defaults it is the shipped decode
(deep/decode.labels_from at cost 4, 1 s / 1 s), checked in sweep.py.

  cost, ha      switching cost; `ha` for a direct change of team when it differs
  power         exponent on the probabilities
  min_team, min_dead   runs shorter than this (s) join the longer neighbour, shortest first
  min_sand      the minimum for a team's run that sits between two runs of the other team
  sand_p        ... and such a run under min_team also needs this peak probability to stay
"""
import numba
import numpy as np


def upsample(p, n):
    step = int(np.ceil(n / len(p)))
    t = np.arange(len(p)) * step
    return np.stack([np.interp(np.arange(n), t, p[:, k]) for k in range(p.shape[1])], 1)


@numba.njit(cache=True)
def viterbi(logp, base):
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    v = logp[0].copy()
    nv = np.empty(3)
    for i in range(1, n):
        for to in range(3):
            best, arg = v[to], to  # staying wins ties, as deep/decode.viterbi
            for fr in range(3):
                s = v[fr] - base[fr, to]
                if s > best:
                    best, arg = s, fr
            back[i, to] = arg
            nv[to] = best + logp[i, to]
        v[:] = nv
    path = np.zeros(n, np.int8)
    path[-1] = np.argmax(v)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


@numba.njit(cache=True)
def absorb(codes, P, min_team, min_dead, min_sand, sand_p):
    """deep/decode.absorb with a second minimum for a team's run between two runs of the other
    team: it stays when it is at least min_sand long and, under min_team, peaks at sand_p."""
    n = len(codes)
    k = np.empty(n, np.int8); m = np.empty(n, np.int64); s = np.empty(n, np.int64)
    r = 0
    for i in range(n):
        if r > 0 and k[r - 1] == codes[i]:
            m[r - 1] += 1
        else:
            k[r] = codes[i]; m[r] = 1; s[r] = i; r += 1
    while r > 1:
        best = -1
        for i in range(r):
            thr = min_dead if k[i] == 2 else min_team
            if k[i] < 2 and m[i] < min_team and m[i] >= min_sand and 0 < i < r - 1 and \
                    k[i - 1] == 1 - k[i] and k[i + 1] == 1 - k[i]:
                if sand_p <= 0 or P[s[i]:s[i] + m[i], k[i]].max() >= sand_p:
                    thr = min_sand
            if m[i] < thr and (best < 0 or m[i] < m[best]):
                best = i
        if best < 0:
            break
        if best == 0:
            j = 1
        elif best == r - 1:
            j = best - 1
        else:
            j = best - 1 if m[best - 1] >= m[best + 1] else best + 1
        if j > best:
            s[j] = s[best]
        m[j] += m[best]
        for i in range(best, r - 1):
            k[i] = k[i + 1]; m[i] = m[i + 1]; s[i] = s[i + 1]
        r -= 1
        w = 0
        for i in range(1, r):
            if k[i] == k[w]:
                m[w] += m[i]
            else:
                w += 1; k[w] = k[i]; m[w] = m[i]; s[w] = s[i]
        r = w + 1
    out = np.empty(n, np.int8)
    for i in range(r):
        out[s[i]:s[i] + m[i]] = k[i]
    return out


def decode_half(P, cost=4.0, ha=None, power=1.0, min_team=1.0, min_dead=1.0, min_sand=None, sand_p=0.0):
    """P: (n, 3) probabilities at 25 fps."""
    lp = power * np.log(np.clip(P, 1e-6, 1))
    base = float(cost) * (1 - np.eye(3))
    if ha is not None:
        base[0, 1] = base[1, 0] = float(ha)
    path = viterbi(lp, base) if cost > 0 else lp.argmax(1).astype(np.int8)
    ms = min_team if min_sand is None else min_sand
    return absorb(path, P, int(round(min_team * 25)), int(round(min_dead * 25)), int(round(ms * 25)), sand_p)


def decode(Ps, spans, n, fn=decode_half, **kw):
    out = np.full(n, 3, np.int8)
    for P, (a, b) in zip(Ps, spans):
        out[a:b + 1] = fn(P, **kw)
    return out


@numba.njit(cache=True)
def insert_touches(path, E, tau, rho, min_len, max_len, edge, min_run):
    """Inside each decoded possession of at least min_run frames, where the other team's
    probability E[:, other] rises to tau or more and falls back below it (at least `edge` frames
    from the possession's ends), write a brief possession of the other team: the frames around the peak
    where E stays above rho * peak, at least min_len and at most max_len frames."""
    out = path.copy()
    n = len(path)
    i = 0
    while i < n:
        j = i
        while j < n and path[j] == path[i]:
            j += 1
        k = path[i]
        if k < 2 and j - i >= min_run:
            o = 1 - k
            t = i + edge
            while t < j - edge:
                if E[t, o] >= tau:
                    # the stretch above tau, and its peak
                    u = t
                    pk = t
                    while u < j - edge and E[u, o] >= tau:
                        if E[u, o] > E[pk, o]:
                            pk = u
                        u += 1
                    if t == i + edge or u >= j - edge:  # a ramp at the possession's end, not a bump inside it
                        t = u + 1
                        continue
                    lo = pk
                    while lo - 1 >= i + 1 and E[lo - 1, o] >= rho * E[pk, o] and pk - (lo - 1) < max_len:
                        lo -= 1
                    hi = pk + 1
                    while hi < j - 1 and E[hi, o] >= rho * E[pk, o] and hi - lo < max_len:
                        hi += 1
                    while hi - lo < min_len:  # grow evenly to the minimum
                        if lo > i + 1:
                            lo -= 1
                        if hi - lo < min_len and hi < j - 1:
                            hi += 1
                        if lo <= i + 1 and hi >= j - 1:
                            break
                    out[lo:hi] = o
                    t = max(u, hi) + edge
                else:
                    t += 1
        i = j
    return out


def decode_insert(P, Pe=None, tau=0.3, rho=0.5, ins_min=0.4, ins_max=3.0, edge=0.4, min_run=2.0, **kw):
    """The plain decode, then brief possessions written in where the evidence Pe peaks."""
    path = decode_half(P, **kw)
    if tau >= 1:
        return path
    Pe = P if Pe is None else Pe
    return insert_touches(path, np.ascontiguousarray(Pe), tau, rho, int(round(ins_min * 25)), int(round(ins_max * 25)),
                          int(round(edge * 25)), int(round(min_run * 25)))
