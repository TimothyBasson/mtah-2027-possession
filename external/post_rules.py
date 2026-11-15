"""Merging rules applied to the decoder's state path, by the context of a brief run.

A run is (state, length, state before, state after). Contexts, with T the other team:
    td      T x D   a brief possession between the other team and a stoppage   -> T
    dt      D x T   a brief possession between a stoppage and the other team   -> T, or -> D
    dd      D x D   a brief possession between two stoppages                   -> D
    tt      T x T   the sandwiched possession (hmm.decode.fold_brief_possessions; rejected)
    d_same  T D T   a brief stoppage inside one team's possession              -> T
    d_diff  T D U   a brief stoppage between the two teams                     -> T (the earlier)
"""

import numpy as np

H, A, D = 0, 1, 2


def context(s, before, after):
    if s == D:
        if before in (H, A) and after in (H, A):
            return "d_same" if before == after else "d_diff"
        return None
    other = 1 - s
    if before == other and after == other:
        return "tt"
    if before == other and after == D:
        return "td"
    if before == D and after == other:
        return "dt"
    if before == D and after == D:
        return "dd"
    return None


def apply_rules(path, limits, dt_to="T"):
    """Copy of one half's path with every run shorter than limits[context] frames rewritten.
    Shortest runs first, repeated until nothing changes, since a rewrite joins its neighbours."""
    path = np.array(path, copy=True)
    if not any(v > 1 for v in limits.values()):
        return path
    while True:
        edges = np.flatnonzero(path[1:] != path[:-1]) + 1
        starts, stops = np.r_[0, edges], np.r_[edges, len(path)]
        best = None
        for i in range(1, len(starts) - 1):
            a, b = starts[i], stops[i]
            c = context(path[a], path[a - 1], path[b])
            if c is not None and b - a < limits.get(c, 0) and (best is None or b - a < best[0]):
                best = (b - a, a, b, c)
        if best is None:
            return path
        _, a, b, c = best
        if c in ("tt", "td", "d_same", "d_diff"):
            path[a:b] = path[a - 1]
        elif c == "dd":
            path[a:b] = D
        elif c == "dt":
            path[a:b] = path[b] if dt_to == "T" else D


def apply_halves(path, halves, limits_seconds, fps, dt_to="T"):
    out = path.copy()
    limits = {k: int(round(v * fps)) for k, v in limits_seconds.items()}
    for lo, hi in halves:
        out[lo:hi] = apply_rules(path[lo:hi], limits, dt_to)
    return out
