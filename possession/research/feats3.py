"""Per-player features for a discrete-choice model of who is on the ball (McFadden's conditional
logit): every tracked player gets a score from hand-made features, and a softmax over the players
on the pitch gives each one's probability of being the player on the ball. The ball is the
training target on the outside matches only; nothing here reads it to predict.

Features are in the player's own team's attacking frame, so the model cannot favour a side.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import STEP, FPS, MAX_SPEED, ROOT

warnings.simplefilter("ignore", RuntimeWarning)
CACHE = os.path.join(ROOT, "data", "cache", "research3")
NAMES = ["speed", "accel", "vx_own", "x_own", "y_edge", "d_all_centroid", "d_own_centroid", "d_opp_centroid",
         "ahead_of_own_centroid", "nn_opp", "nn_mate", "opp_in3", "opp_in6", "opp_in10", "mate_in5", "mate_in10",
         "press", "close", "approach_all", "d_converge", "is_keeper", "depth_rank", "toward_centroid",
         "nn_opp_sq", "speed_sq", "d_all_sq"]


def compute(players, ball, team, halves, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    parts, targets, idxs = [], [], []
    K = players.shape[1]
    for a, b in halves:
        idx = np.arange(a, b + 1, STEP)
        xy = players[idx, :, :2].astype(float)
        lo, hi = np.clip(idx - STEP, a, b), np.clip(idx + STEP, a, b)
        v = (players[hi, :, :2].astype(float) - players[lo, :, :2]) * FPS / (hi - lo)[:, None, None]
        sp = np.hypot(v[..., 0], v[..., 1])
        v[sp > MAX_SPEED] = np.nan
        sp = np.hypot(v[..., 0], v[..., 1])
        ok = np.isfinite(xy[..., 0])
        n = len(idx)
        home_left = np.nanmean(xy[:, team == 0, 0]) < np.nanmean(xy[:, team == 1, 0])
        sign = np.where(team == 0, 1.0, -1.0) * (1 if home_left else -1)     # +1: this slot's team attacks +x
        x_own = np.where(sign > 0, xy[..., 0], L - xy[..., 0])
        vx_own = v[..., 0] * sign
        f = np.full((n, K, len(NAMES)), np.nan, np.float32)
        put = lambda name, val: f.__setitem__((slice(None), slice(None), NAMES.index(name)), val)
        acc = np.abs(np.gradient(np.nan_to_num(sp), axis=0)) * (FPS / STEP)
        put("speed", sp); put("accel", np.clip(acc, 0, 8)); put("vx_own", vx_own); put("x_own", x_own)
        put("y_edge", np.minimum(xy[..., 1], W - xy[..., 1]))
        call = np.nanmean(xy, 1)
        d_all = np.linalg.norm(xy - call[:, None], axis=-1)
        put("d_all_centroid", d_all); put("d_all_sq", d_all ** 2 / 100)
        cen = {t: np.nanmean(xy[:, team == t], 1) for t in (0, 1)}
        own_c = np.where((team == 0)[None, :, None], cen[0][:, None], cen[1][:, None])
        opp_c = np.where((team == 0)[None, :, None], cen[1][:, None], cen[0][:, None])
        put("d_own_centroid", np.linalg.norm(xy - own_c, axis=-1))
        put("d_opp_centroid", np.linalg.norm(xy - opp_c, axis=-1))
        put("ahead_of_own_centroid", (xy[..., 0] - own_c[..., 0]) * sign)
        d = xy[:, :, None, :] - xy[:, None, :, :]                  # (n, i, j, 2): from j to i
        dist = np.linalg.norm(d, axis=-1)
        pair_ok = ok[:, :, None] & ok[:, None, :] & ~np.eye(K, dtype=bool)[None]
        opp = (team[:, None] != team[None, :])[None] & pair_ok
        mate = (team[:, None] == team[None, :])[None] & pair_ok
        dopp = np.where(opp, dist, np.inf); dmate = np.where(mate, dist, np.inf)
        nn_opp = np.clip(dopp.min(2), 0, 30)
        put("nn_opp", nn_opp); put("nn_opp_sq", nn_opp ** 2 / 100); put("nn_mate", np.clip(dmate.min(2), 0, 30))
        for r in (3, 6, 10):
            put(f"opp_in{r}", (dopp < r).sum(2))
        for r in (5, 10):
            put(f"mate_in{r}", (dmate < r).sum(2))
        vj = np.nan_to_num(v)
        toward = (vj[:, None, :, :] * d).sum(-1) / (dist + 1e-6)       # j's speed toward i
        rel = ((vj[:, None, :, :] - vj[:, :, None, :]) * d).sum(-1) / (dist + 1e-6)
        w = np.exp(-dist / 5.0)
        put("press", np.where(opp, np.maximum(toward, 0) * w, 0).sum(2))
        put("close", np.where(opp, np.maximum(rel, 0) * w, 0).sum(2))
        put("approach_all", np.where(pair_ok, toward * np.exp(-dist / 15.0), 0).sum(2))
        # motion-field convergence point of everybody
        spz = np.nan_to_num(sp)
        ww = np.where(ok & (spz > 1.0), spz, 0.0)
        u = vj / (spz[..., None] + 1e-9)
        M = ww[..., None, None] * (np.eye(2) - u[..., :, None] * u[..., None, :])
        A = M.sum(1) + 1e-3 * np.eye(2)
        p = np.linalg.solve(A, (M @ np.nan_to_num(xy)[..., None])[..., 0].sum(1)[..., None])[..., 0]
        put("d_converge", np.clip(np.linalg.norm(xy - p[:, None], axis=-1), 0, 40))
        depth = np.where(ok, x_own, np.inf)
        rank = np.zeros((n, K))
        for t in (0, 1):
            cols = np.flatnonzero(team == t)
            rank[:, cols] = np.argsort(np.argsort(depth[:, cols], 1), 1)
        put("is_keeper", (rank == 0).astype(float)); put("depth_rank", np.clip(rank, 0, 10) / 10)
        tc = call[:, None] - xy
        put("toward_centroid", (vj * tc).sum(-1) / (np.linalg.norm(tc, axis=-1) + 1e-6))
        sq = np.nan_to_num(sp)
        put("speed_sq", sq ** 2 / 10)
        f[~ok] = np.nan
        # target: the player nearest the ball, if within 2 m and the ball is seen
        bd = np.linalg.norm(xy - ball[idx, None, :2].astype(float), axis=-1)
        bd = np.where(ok, bd, np.inf)
        tgt = np.where(bd.min(1) < 2.0, bd.argmin(1), -1)
        parts.append(f); targets.append(tgt); idxs.append(idx)
    return np.concatenate(parts), np.concatenate(targets), np.concatenate(idxs)


def cached(path):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, os.path.basename(path))
    if os.path.exists(out):
        z = np.load(out)
        return z["f"], z["target"], z["team"]
    d = np.load(path)
    f, tgt, idx = compute(d["players"], d["ball"], d["team"], [tuple(h) for h in d["halves"]], d["pitch"])
    np.savez_compressed(out, f=f, target=tgt, team=d["team"])
    return f, tgt, d["team"]


if __name__ == "__main__":
    import glob
    from concurrent.futures import ProcessPoolExecutor
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    with ProcessPoolExecutor(3) as ex:
        for p, r in zip(paths, ex.map(cached, paths)):
            print(os.path.basename(p), r[0].shape, round(float((r[1] >= 0).mean()), 2), flush=True)
