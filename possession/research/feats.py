"""Players-only features at 5 Hz for the research harness (possession/research/README.md).

Every "who" feature is antisymmetric: home's value in home's attacking frame minus away's in
away's, so swapping the teams flips its sign and a model on them cannot favour a side.
"""
import os
import sys
import warnings

import numpy as np

STEP = 5          # 25 fps -> 5 Hz
FPS = 25
HZ = FPS // STEP
MAX_SPEED = 12.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data", "cache", "research")

warnings.simplefilter("ignore", RuntimeWarning)


def _team_block(xy, v, L):
    """Per-team shape and motion in its own attacking frame (attacks +x). xy (n,k,2), v (n,k,2)."""
    n, k, _ = xy.shape
    ok = np.isfinite(xy[..., 0])
    depth = np.where(ok, xy[..., 0], np.inf)
    keeper = depth.argmin(1)
    out_ok = ok.copy()
    out_ok[np.arange(n), keeper] = False          # outfield players
    x = np.where(out_ok, xy[..., 0], np.nan)
    y = np.where(out_ok, xy[..., 1], np.nan)
    cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
    f = {}
    f["spread"] = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1)
    f["wid"] = np.nanstd(y, 1)
    f["len"] = np.nanstd(x, 1)
    f["cx"] = cx
    xs = np.sort(np.where(out_ok, xy[..., 0], np.inf), 1)
    f["back"] = np.where(np.isfinite(xs[:, 0]), xs[:, 0], np.nan)
    f["back2"] = np.where(np.isfinite(xs[:, 1]), xs[:, 1], np.nan)
    f["front"] = np.nanmax(x, 1)
    vx = np.where(out_ok, v[..., 0], np.nan)
    vy = np.where(out_ok, v[..., 1], np.nan)
    sp = np.hypot(vx, vy)
    f["vcx"] = np.nanmean(vx, 1)
    f["speed"] = np.nanmean(sp, 1)
    f["speed_sd"] = np.nanstd(sp, 1)
    mv = sp > 0.5
    ux, uy = np.where(mv, vx / sp, np.nan), np.where(mv, vy / sp, np.nan)
    f["pol"] = np.hypot(np.nanmean(ux, 1), np.nanmean(uy, 1))        # all moving the same way = 1
    # expansion: mean radial speed away from the team's centre (positive = stretching)
    rx, ry = x - cx[:, None], y - cy[:, None]
    r = np.hypot(rx, ry) + 1e-6
    f["expand"] = np.nanmean((vx - np.nanmean(vx, 1)[:, None]) * rx / r + (vy - np.nanmean(vy, 1)[:, None]) * ry / r, 1)
    f["fwd_runs"] = np.nanmean(np.where(out_ok, vx > 2.0, np.nan), 1)   # share sprinting forward
    f["back_runs"] = np.nanmean(np.where(out_ok, vx < -2.0, np.nan), 1)
    f["n"] = out_ok.sum(1).astype(float)
    return f, out_ok


def _cross(xh, okh, xa, oka):
    """Nearest-opponent distances. xh, xa in the HOME frame."""
    d = np.linalg.norm(xh[:, :, None, :] - xa[:, None, :, :], axis=-1)   # (n, kh, ka)
    d = np.where(okh[:, :, None] & oka[:, None, :], d, np.inf)
    nh, na = d.min(2), d.min(1)                                          # each player's nearest opponent
    return np.where(np.isfinite(nh), nh, np.nan), np.where(np.isfinite(na), na, np.nan)


def _by_depth(vals, depth, ok, lo, hi):
    """Mean of vals over each team's lo..hi-th deepest outfield players."""
    order = np.argsort(np.where(ok, depth, np.inf), 1)[:, lo:hi]
    v = np.take_along_axis(vals, order, 1)
    o = np.take_along_axis(ok, order, 1)
    return np.nanmean(np.where(o, v, np.nan), 1)


def _voronoi(xh, okh, xa, oka, L, W, cell=5.0):
    """Share of the pitch (and of each half) nearest to a home player, home frame."""
    gx, gy = np.meshgrid(np.arange(cell / 2, L, cell), np.arange(cell / 2, W, cell))
    g = np.stack([gx.ravel(), gy.ravel()], -1)                           # (G, 2)
    n = len(xh)
    tot, opp, own = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    far = g[:, 0] > L / 2
    for s in range(0, n, 2000):
        e = min(n, s + 2000)
        dh = np.linalg.norm(g[None, :, None, :] - np.where(okh[s:e, None, :, None], xh[s:e, None], 1e6), axis=-1).min(2)
        da = np.linalg.norm(g[None, :, None, :] - np.where(oka[s:e, None, :, None], xa[s:e, None], 1e6), axis=-1).min(2)
        hw = (dh < da).astype(float)
        hw[(dh > 1e5) | (da > 1e5)] = np.nan
        tot[s:e], opp[s:e], own[s:e] = np.nanmean(hw, 1), np.nanmean(hw[:, far], 1), np.nanmean(hw[:, ~far], 1)
    return tot, opp, own


def _converge(x, v, ok):
    """Motion-field convergence (Kim et al., CVPR 2010): the point the moving players' velocity
    lines pass closest to, weighted by speed. x, v (n,k,2) home frame."""
    sp = np.hypot(v[..., 0], v[..., 1])
    w = np.where(ok & (sp > 1.0), sp, 0.0)
    d = np.nan_to_num(v / (sp[..., None] + 1e-9))
    xs = np.nan_to_num(x)
    # sum_i w_i (I - d d^T) (p - x_i) = 0
    A = np.zeros((len(x), 2, 2)); b = np.zeros((len(x), 2))
    I = np.eye(2)
    M = w[..., None, None] * (I - d[..., :, None] * d[..., None, :])
    A = M.sum(1) + 1e-3 * I
    b = (M @ xs[..., None])[..., 0].sum(1)
    p = np.linalg.solve(A, b[..., None])[..., 0]
    p[(w > 0).sum(1) < 4] = np.nan
    return p


def compute(players, team, halves, pitch):
    """dict(idx=frames sampled, who={name: (n,)}, dead={name: (n,)}, half=(n,) half number)."""
    L, W = float(pitch[0]), float(pitch[1])
    who, dead, idxs, halfno = {}, {}, [], []
    parts = []
    for hi, (a, b) in enumerate(halves):
        idx = np.arange(a, b + 1, STEP)
        xy = players[idx, :, :2].astype(float)
        lo, hi2 = np.clip(idx - STEP, a, b), np.clip(idx + STEP, a, b)
        v = (players[hi2, :, :2].astype(float) - players[lo, :, :2]) * FPS / (hi2 - lo)[:, None, None]
        sp = np.hypot(v[..., 0], v[..., 1])
        v[sp > MAX_SPEED] = np.nan
        # home frame: home attacks +x
        home_left = np.nanmean(xy[:, team == 0, 0]) < np.nanmean(xy[:, team == 1, 0])
        if not home_left:
            xy = np.array([L, W]) - xy
            v = -v
        xh, xa, vh, va = xy[:, team == 0], xy[:, team == 1], v[:, team == 0], v[:, team == 1]
        flip = lambda p: np.array([L, W]) - p
        fh, okh = _team_block(xh, vh, L)
        fa, oka = _team_block(flip(xa), -va, L)
        f = {k: fh[k] - fa[k] for k in fh if k != "n"}
        nh, na = _cross(xh, okh, xa, oka)
        f["nn_opp"] = np.nanmean(nh, 1) - np.nanmean(na, 1)
        f["nn_opp_back"] = _by_depth(nh, xh[..., 0], okh, 0, 4) - _by_depth(na, L - xa[..., 0], oka, 0, 4)
        f["nn_opp_front"] = _by_depth(nh, -xh[..., 0], okh, 0, 3) - _by_depth(na, -(L - xa[..., 0]), oka, 0, 3)
        f["free"] = np.nanmean(np.where(okh, nh > 5, np.nan), 1) - np.nanmean(np.where(oka, na > 5, np.nan), 1)
        vt, vo, vw = _voronoi(xh, okh, xa, oka, L, W)
        f["vor"] = vt - 0.5
        f["vor_opp"] = vo - (1 - vw)       # home's share of away's half minus away's share of home's half
        okall = np.concatenate([okh, oka], 1)
        xall, vall = np.concatenate([xh, xa], 1), np.concatenate([vh, va], 1)
        p = _converge(xall, vall, okall)
        dh = np.where(okh, np.linalg.norm(xh - p[:, None], axis=-1), np.inf).min(1)
        da = np.where(oka, np.linalg.norm(xa - p[:, None], axis=-1), np.inf).min(1)
        f["conv_near"] = np.where(np.isfinite(dh) & np.isfinite(da), np.clip(da - dh, -15, 15), np.nan)
        f["conv_x"] = np.clip(p[:, 0], -20, L + 20) - L / 2
        # field position of everybody, home frame
        allx = np.where(okall, xall[..., 0], np.nan)
        f["field_x"] = np.nanmean(allx, 1) - L / 2
        f["vx_all"] = np.nanmean(np.where(okall, vall[..., 0], np.nan), 1)
        # dead-ball (symmetric) signals
        spall = np.where(np.isfinite(xall[..., 0]), np.hypot(vall[..., 0], vall[..., 1]), np.nan)
        g = {}
        g["sp_med"] = np.nanmedian(spall, 1)
        g["sp_p80"] = np.nanpercentile(spall, 80, 1)
        g["sp_max"] = np.nanmax(spall, 1)
        g["still"] = np.nanmean(np.where(np.isfinite(spall), spall < 1.0, np.nan), 1)
        yall = xall[..., 1]
        edge = np.minimum(yall, W - yall)
        g["touch_min"] = np.clip(np.nanmin(np.where(np.isfinite(yall), edge, np.nan), 1), -3, 10)   # throw-in taker
        gl = np.minimum(xall[..., 0], L - xall[..., 0])
        corner = np.hypot(gl, edge)
        g["corner_min"] = np.clip(np.nanmin(corner, 1), 0, 30)
        inbox = (gl < 16.5) & (np.abs(yall - W / 2) < 20.16)
        okf = np.isfinite(xall[..., 0])
        left = xall[..., 0] < L / 2
        g["box_max"] = np.maximum((inbox & left & okf).sum(1), (inbox & ~left & okf).sum(1)).astype(float)
        g["own_half"] = (np.nanmean(np.where(okh, xh[..., 0] < L / 2 + 1, np.nan), 1)
                         + np.nanmean(np.where(oka, xa[..., 0] > L / 2 - 1, np.nan), 1)) / 2   # kickoff shape
        g["spread_sum"] = fh["spread"] + fa["spread"]
        g["len_sum"] = fh["len"] + fa["len"]
        g["pol_sum"] = fh["pol"] + fa["pol"]
        g["nn_opp_all"] = (np.nanmean(nh, 1) + np.nanmean(na, 1)) / 2
        g["sp_sd"] = np.nanstd(spall, 1)
        g["n_trk"] = okf.sum(1).astype(float)
        # per-player |acceleration| 80th percentile
        accs = np.abs(np.gradient(np.nan_to_num(spall, nan=0.0), axis=0)) * HZ
        accs[~np.isfinite(spall)] = np.nan
        accs[accs > 8] = np.nan
        g["acc_p80"] = np.nanpercentile(accs, 80, 1)
        parts.append((idx, np.full(len(idx), hi), f, g))
    idx = np.concatenate([p[0] for p in parts])
    half = np.concatenate([p[1] for p in parts])
    who = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
    dead = {k: np.concatenate([p[3][k] for p in parts]) for k in parts[0][3]}
    return dict(idx=idx, half=half, who=who, dead=dead)


def cached(path):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, os.path.basename(path))
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        return {k: (z[k].item() if z[k].dtype == object else z[k]) for k in z}
    d = np.load(path)
    r = compute(d["players"], d["team"], [tuple(h) for h in d["halves"]], d["pitch"])
    r["labels"] = d["labels"][r["idx"]]
    r["labels25"] = d["labels"]
    r["halves"] = d["halves"]
    np.savez_compressed(out, **{k: (np.array(v, dtype=object) if isinstance(v, dict) else v) for k, v in r.items()})
    return r


if __name__ == "__main__":
    import glob
    from concurrent.futures import ProcessPoolExecutor
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    with ProcessPoolExecutor(4) as ex:
        for p, r in zip(paths, ex.map(cached, paths)):
            print(os.path.basename(p), len(r["idx"]), flush=True)
