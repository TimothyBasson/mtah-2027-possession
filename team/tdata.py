"""Tensors for the team-naming net: deep/data.py's players-only set per frame, plus each player's
own path over the second either side. Stage 1 hands the set encoder a player's position and one
velocity; here the encoder also sees how that player has just moved and is about to move (the
checks, turns and bursts around a change of possession), before the players are pooled.

Still players only. The ball is read as a training target, and the folded labels come from the
training matches' ball (external/control.py); neither is an input.
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import data as D  # noqa: E402  (deep/data.py)

OFFSETS = (-25, -12, -6, -3, 3, 6, 12, 25)  # frames at 25 fps: 0.12 s to 1 s either side
NFEAT = D.NFEAT + 3 * len(OFFSETS)           # base 8, then (dx, dy) per offset, then a seen flag per offset
PATH = slice(D.NFEAT, D.NFEAT + 2 * len(OFFSETS))


def half_tensors(players, ball, team, a, b, pitch, fps=25):
    """deep/data.half_tensors with the path features appended, slot order kept the same."""
    assert fps == 25
    X, M, bxy, fr = D.half_tensors(players, ball, team, a, b, pitch, fps)
    L, W = pitch
    xy = players[a:b + 1, :, :2].astype(np.float32) - np.array([L / 2, W / 2], np.float32)
    if not D.home_defends_left(players, team, a, b):
        xy = -xy
    n, K = xy.shape[:2]
    idx = np.arange(0, n, D.STEP)
    ok = np.isfinite(xy[idx, :, 0])
    extra = np.zeros((len(idx), K, 3 * len(OFFSETS)), np.float32)
    for j, k in enumerate(OFFSETS):
        t = idx + k
        inside = (t >= 0) & (t < n)
        d = np.full((len(idx), K, 2), np.nan, np.float32)
        d[inside] = (xy[t[inside]] - xy[idx[inside]]) / (abs(k) / fps)   # m/s over the span, signed in time
        good = np.isfinite(d[..., 0]) & (np.linalg.norm(d, axis=-1) <= D.MAX_SPEED)
        extra[..., 2 * j:2 * j + 2] = np.where(good[..., None], d / 5, 0)
        extra[..., 2 * len(OFFSETS) + j] = good
    extra[~ok] = 0
    order = np.argsort(~ok, axis=1, kind="stable")[:, :D.SLOTS]  # the order deep/data.py used
    E = np.take_along_axis(extra, order[:, :, None], 1)
    if K < D.SLOTS:
        E = np.pad(E, ((0, 0), (0, D.SLOTS - K), (0, 0)))
    return np.concatenate([X, E], -1), M, bxy, fr


def load_match(path):
    d = np.load(path)
    name = os.path.basename(path)[:-4]
    raw, fold, spans = C.truth(name)
    out = []
    for a, b in spans:
        X, M, bxy, fr = half_tensors(d["players"], d["ball"], d["team"], a, b, tuple(d["pitch"]), int(d["fps"]))
        code = lambda lab: np.select([lab[fr] == c for c in "HAD"], [0, 1, 2], -1).astype(np.int64)  # noqa: E731
        out.append(dict(X=X.astype(np.float16), M=M, y=code(fold), yraw=code(raw), ball=bxy, frames=fr))
    return dict(name=name, provider=D.provider(path), halves=out, spans=spans)


def load_all():
    cache = os.path.join(C.ROOT, "data", "cache", "team", "tensors.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    from multiprocessing import Pool
    with Pool(3) as pool:
        ms = pool.map(load_match, C.paths())
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache + ".tmp", "wb") as f:
        pickle.dump(ms, f, protocol=4)
    os.replace(cache + ".tmp", cache)
    return ms


def swap_teams(X):
    """The same play seen from the other team: half a turn of the pitch, home <-> away."""
    X = X.copy()
    X[..., :4] *= -1
    X[..., 6] *= -1
    X[..., PATH] *= -1
    return X


def mirror(X):
    """Mirror across the long axis (y -> -y)."""
    X = X.copy()
    X[..., 1] *= -1
    X[..., 3] *= -1
    X[..., D.NFEAT + 1:D.NFEAT + 2 * len(OFFSETS):2] *= -1
    return X
