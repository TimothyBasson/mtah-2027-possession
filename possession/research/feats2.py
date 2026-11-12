"""Second batch of 'who' features: lines, keeper, and player-level pressure. Same conventions as feats.py."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats
from feats import STEP, FPS, MAX_SPEED, ROOT

warnings.simplefilter("ignore", RuntimeWarning)
CACHE = os.path.join(ROOT, "data", "cache", "research2")


def _lines(xy, v):
    """xy, v in the team's own frame (attacks +x)."""
    n = len(xy)
    ok = np.isfinite(xy[..., 0])
    depth = np.where(ok, xy[..., 0], np.inf)
    order = np.argsort(depth, 1)
    take = lambda a, sl: np.take_along_axis(a, order[:, sl], 1)
    f = {}
    kx, ky = take(xy[..., 0], slice(0, 1))[:, 0], take(xy[..., 1], slice(0, 1))[:, 0]
    f["keeper_x"] = np.clip(kx, -5, 60)
    f["keeper_speed"] = np.hypot(take(v[..., 0], slice(0, 1))[:, 0], take(v[..., 1], slice(0, 1))[:, 0])
    bx, by = take(xy[..., 0], slice(1, 5)), take(xy[..., 1], slice(1, 5))
    bx = np.where(np.isfinite(bx), bx, np.nan)
    f["back4_wid"] = np.nanstd(by, 1)
    f["back4_flat"] = np.nanstd(bx, 1)
    f["keeper_gap"] = np.nanmean(bx, 1) - kx
    f["back4_speed"] = np.nanmean(np.hypot(take(v[..., 0], slice(1, 5)), take(v[..., 1], slice(1, 5))), 1)
    f["back4_vx"] = np.nanmean(take(v[..., 0], slice(1, 5)), 1)
    cnt = ok.sum(1)
    rev = np.argsort(np.where(ok, -xy[..., 0], np.inf), 1)
    fx, fy = np.take_along_axis(xy[..., 0], rev[:, :3], 1), np.take_along_axis(xy[..., 1], rev[:, :3], 1)
    f["front3_wid"] = np.nanstd(fy, 1)
    f["front3_vx"] = np.nanmean(np.take_along_axis(v[..., 0], rev[:, :3], 1), 1)
    f["front3_speed"] = np.nanmean(np.hypot(np.take_along_axis(v[..., 0], rev[:, :3], 1), np.take_along_axis(v[..., 1], rev[:, :3], 1)), 1)
    # widest players: how close the two widest stand to the touchlines (attacking teams hold width)
    y = np.where(ok, xy[..., 1], np.nan)
    f["yrange"] = np.nanmax(y, 1) - np.nanmin(y, 1)
    f["xrange"] = np.nanmax(np.where(ok, xy[..., 0], np.nan), 1) - bx[:, 0]
    return f


def _pressure(xt, vt, okt, xo, vo, oko):
    """How hard opponents close on each player of team t: sum over opponents of their speed toward
    him, discounted by distance (5 m scale). Returns per-team max, sum, and closing on the most
    pressed player's neighbourhood."""
    d = xt[:, :, None, :] - xo[:, None, :, :]                 # from opponent j to player i
    dist = np.linalg.norm(d, axis=-1) + 1e-6
    toward = (np.nan_to_num(vo)[:, None, :, :] * d).sum(-1) / dist
    rel = ((np.nan_to_num(vo)[:, None, :, :] - np.nan_to_num(vt)[:, :, None, :]) * d).sum(-1) / dist   # closing speed
    w = np.exp(-dist / 5.0)
    m = okt[:, :, None] & oko[:, None, :]
    P = np.where(m, np.maximum(toward, 0) * w, 0).sum(2)
    C = np.where(m, np.maximum(rel, 0) * w, 0).sum(2)
    near = np.where(m, dist < 3.0, False).sum(2)
    return P.max(1), P.sum(1), C.max(1), C.sum(1), near.max(1).astype(float)


def compute(players, team, halves, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    parts = []
    for hi, (a, b) in enumerate(halves):
        idx = np.arange(a, b + 1, STEP)
        xy = players[idx, :, :2].astype(float)
        lo, hi2 = np.clip(idx - STEP, a, b), np.clip(idx + STEP, a, b)
        v = (players[hi2, :, :2].astype(float) - players[lo, :, :2]) * FPS / (hi2 - lo)[:, None, None]
        v[np.hypot(v[..., 0], v[..., 1]) > MAX_SPEED] = np.nan
        if not (np.nanmean(xy[:, team == 0, 0]) < np.nanmean(xy[:, team == 1, 0])):
            xy = np.array([L, W]) - xy; v = -v
        xh, xa, vh, va = xy[:, team == 0], xy[:, team == 1], v[:, team == 0], v[:, team == 1]
        fh, fa = _lines(xh, vh), _lines(np.array([L, W]) - xa, -va)
        f = {k: fh[k] - fa[k] for k in fh}
        okh, oka = np.isfinite(xh[..., 0]), np.isfinite(xa[..., 0])
        ph, pa = _pressure(xh, vh, okh, xa, va, oka), _pressure(xa, va, oka, xh, vh, okh)
        for nm, x, y in zip(("press_max", "press_sum", "close_max", "close_sum", "crowd_max"), ph, pa):
            f[nm] = x - y        # positive: home players are the ones being closed down
        parts.append(f)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def cached(path):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, os.path.basename(path))
    if os.path.exists(out):
        z = np.load(out)
        return {k: z[k] for k in z}
    d = np.load(path)
    r = compute(d["players"], d["team"], [tuple(h) for h in d["halves"]], d["pitch"])
    np.savez_compressed(out, **r)
    return r


if __name__ == "__main__":
    import glob
    from concurrent.futures import ProcessPoolExecutor
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    with ProcessPoolExecutor(4) as ex:
        for p, r in zip(paths, ex.map(cached, paths)):
            pass
    print("done", len(paths))
