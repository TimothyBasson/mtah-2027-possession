"""Tensors for the turnover detector. Players only as input; the ball and the labels are read
here solely to build training targets (and, in scoring, the yardstick).

Unlike deep/data.py the players keep their column (identity) through a half, at 12.5 fps, so the
detector can follow each player's path in time: a tackle or an interception is something one or
two players do over half a second, which per-frame team pooling at 5 fps blurs.
"""

import os
import pickle
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external")]
import data as D  # noqa: E402

STEP = 2      # 25 fps -> 12.5 fps
LAG = 3       # frames either side for velocity and acceleration (0.24 s in all)
NFEAT = 9     # x, y, vx, vy, ax, ay, speed, team, has-velocity
K = 26        # players kept per window
CACHE = os.path.join(ROOT, "data", "cache", "turnover_v1.pkl")


def half_inputs(players, team, a, b, pitch, fps=25):
    """One half -> (X (T, k, NFEAT) float16, M (T, k) bool, cols, frames, flip). Players only."""
    L, W = pitch
    xy = players[a:b + 1, :, :2].astype(np.float32).copy()
    sp = players[a:b + 1, :, 2].astype(np.float32).copy()
    flip = not D.home_defends_left(players, team, a, b)
    xy -= np.array([L / 2, W / 2], np.float32)
    if flip:
        xy = -xy
    sp[sp > D.MAX_SPEED] = np.nan
    n = len(xy)
    vel = np.full_like(xy, np.nan)
    vel[LAG:n - LAG] = (xy[2 * LAG:] - xy[:n - 2 * LAG]) * (fps / (2 * LAG))
    acc = np.full_like(xy, np.nan)
    acc[LAG:n - LAG] = (xy[2 * LAG:] - 2 * xy[LAG:n - LAG] + xy[:n - 2 * LAG]) * (fps / LAG) ** 2
    bad = ~(np.linalg.norm(vel, axis=-1) <= D.MAX_SPEED)
    vel[bad] = np.nan
    acc[bad] = np.nan
    idx = np.arange(0, n, STEP)
    ok = np.isfinite(xy[idx, :, 0])
    cols = np.flatnonzero(ok.any(0))
    xy, sp, vel, acc, ok = xy[idx][:, cols], sp[idx][:, cols], vel[idx][:, cols], acc[idx][:, cols], ok[:, cols]
    T, k = ok.shape
    hasv = np.isfinite(vel[:, :, 0])
    feat = np.stack([xy[:, :, 0] / 52.5, xy[:, :, 1] / 34.0,
                     np.nan_to_num(vel[:, :, 0]) / 5, np.nan_to_num(vel[:, :, 1]) / 5,
                     np.clip(np.nan_to_num(acc[:, :, 0]), -12, 12) / 4, np.clip(np.nan_to_num(acc[:, :, 1]), -12, 12) / 4,
                     np.nan_to_num(sp) / 5,
                     np.broadcast_to(np.where(team[cols] == 0, 1.0, -1.0), (T, k)),
                     hasv.astype(np.float32)], -1)
    feat[~ok] = 0
    return feat.astype(np.float16), ok, cols, a + idx, flip


def events_at(labels, frames):
    """Per kept frame: 0 nothing, 1 the ball passes to home here, 2 to away (a direct H<->A change
    of the labels, placed on the nearest kept frame)."""
    l = np.asarray(labels)
    ev = np.zeros(len(frames), np.int8)
    tv = np.flatnonzero(((l[1:] == "H") & (l[:-1] == "A")) | ((l[1:] == "A") & (l[:-1] == "H"))) + 1
    tv = tv[(tv >= frames[0]) & (tv <= frames[-1])]
    ev[np.clip(np.round((tv - frames[0]) / STEP).astype(int), 0, len(frames) - 1)] = np.where(l[tv] == "H", 1, 2)
    return ev


def load_match(path):
    """Inputs and training targets of one outside match."""
    import score_hmm as sh
    from control import control_labels
    from hmm.ball import detect_touches
    d = sh.load(path)
    players, ball, team = d["players"], d["ball"].astype(float), d["team"]
    lab, fold = d["labels"], control_labels(d)
    code = np.full(len(lab), -1, np.int8)
    for i, c in enumerate("HAD"):
        code[lab == c] = i
    dist = np.hypot(players[:, :, 0].astype(float) - ball[:, None, 0], players[:, :, 1].astype(float) - ball[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dh = np.nanmin(dist[:, team == 0], 1)
        da = np.nanmin(dist[:, team == 1], 1)
    t = detect_touches(ball[:, 0], ball[:, 1], dh, da, int(d["fps"]))
    halves = []
    for a, b in [tuple(int(v) for v in h) for h in d["halves"]]:
        X, M, cols, fr, flip = half_inputs(players, team, a, b, tuple(d["pitch"]), int(d["fps"]))
        bxy = ball[fr, :2] - np.array(d["pitch"], float) / 2
        if flip:
            bxy = -bxy
        touch = np.full(len(fr), -1, np.int8)  # column (within cols) of the player touching the ball
        for f, side in zip(t["frame"], t["team"]):
            if a <= f <= b and side >= 0:
                dd = dist[f, cols]
                dd = np.where(team[cols] == side, dd, np.inf)
                if np.isfinite(dd).any():
                    touch[min((f - a) // STEP, len(fr) - 1)] = int(np.nanargmin(dd))
        halves.append(dict(X=X, M=M, y=code[fr], ev=events_at(lab, fr), evf=events_at(fold, fr),
                           ball=(bxy / np.array([52.5, 34.0])).astype(np.float32), touch=touch, frames=fr))
    return dict(name=os.path.basename(path)[:-4], provider=D.provider(path), halves=halves)


def load_all():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from multiprocessing import Pool
    with Pool(3) as pool:
        ms = pool.map(load_match, D.external_paths())
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE + ".tmp", "wb") as f:
        pickle.dump(ms, f, protocol=4)
    os.replace(CACHE + ".tmp", CACHE)
    return ms


def window(h, s, T, rng=None, drop=0.0):
    """T frames of a half from s: the K most-tracked players of the window, their columns kept."""
    X, M = h["X"][s:s + T], h["M"][s:s + T]
    cnt = M.sum(0)
    keep = np.argsort(-cnt, kind="stable")[:K]
    keep = np.sort(keep[cnt[keep] > 0])
    Xo = np.zeros((T, K, NFEAT), np.float32)
    Mo = np.zeros((T, K), bool)
    Xo[:len(X), :len(keep)] = X[:, keep]
    Mo[:len(X), :len(keep)] = M[:, keep]
    touch = np.full(T, -1, np.int64)
    if "touch" in h:
        pos = np.full(h["M"].shape[1], -1)
        pos[keep] = np.arange(len(keep))
        tt = h["touch"][s:s + T]
        touch[:len(tt)] = np.where(tt >= 0, pos[np.clip(tt, 0, None)], -1)
    return Xo, Mo, touch


def swap_teams(X):
    X = X.copy()
    X[..., :6] *= -1
    X[..., 7] *= -1
    return X


def mirror(X):
    X = X.copy()
    X[..., [1, 3, 5]] *= -1
    return X


if __name__ == "__main__":
    ms = load_all()
    for m in ms[:1] + ms[-1:]:
        h = m["halves"][0]
        print(m["name"], h["X"].shape, "events/min", (h["ev"] > 0).sum() / (len(h["y"]) / 750), "folded", (h["evf"] > 0).sum() / (len(h["y"]) / 750),
              "touches/min", (h["touch"] >= 0).sum() / (len(h["y"]) / 750))
