"""Tracking arrays -> tensors for the learned possession model. Players only: the ball is read
here solely as a training target (the auxiliary "where is the ball" head), never as an input.

Each half becomes a sequence at STEP-frame spacing (5 fps), with a fixed-size, order-free set of
players per frame. The pitch is turned so home always defends x < 0, which lets one model serve
both halves; swapping the teams and mirroring the pitch are then exact symmetries used as
augmentation in training.
"""

import glob
import os
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP = 5          # 25 fps -> 5 fps
LAG = 5           # frames either side for velocity and change of speed (0.4 s in all)
MAX_SPEED = 12.0  # m/s; faster is a track jumping, as possession/model.py
SLOTS = 26        # players kept per frame (22 expected; trackers sometimes duplicate one)
NFEAT = 8         # x, y, vx, vy, speed, change of speed, team (+1 home, -1 away), has-velocity
CLASSES = "HAD"


def provider(path):
    return os.path.basename(path).split("_")[0]


def external_paths():
    return sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))


def half_tensors(players, ball, team, a, b, pitch, fps=25):
    """One half -> (X (T, SLOTS, NFEAT), mask (T, SLOTS), ball (T, 2), frames (T,)), canonical frame."""
    L, W = pitch
    xy = players[a:b + 1, :, :2].astype(np.float32).copy()
    sp = players[a:b + 1, :, 2].astype(np.float32).copy()
    bxy = ball[a:b + 1, :2].astype(np.float32).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_left = np.nanmean(xy[:, team == 0, 0]) < L / 2
    xy -= np.array([L / 2, W / 2], np.float32)
    bxy -= np.array([L / 2, W / 2], np.float32)
    if not home_left:  # turn the pitch half a turn
        xy, bxy = -xy, -bxy
    sp[sp > MAX_SPEED] = np.nan
    n = len(xy)
    vel = np.full_like(xy, np.nan)
    vel[LAG:n - LAG] = (xy[2 * LAG:] - xy[:n - 2 * LAG]) * (fps / (2 * LAG))
    vel[np.linalg.norm(vel, axis=-1) > MAX_SPEED] = np.nan
    dsp = np.full_like(sp, np.nan)
    dsp[LAG:n - LAG] = (sp[2 * LAG:] - sp[:n - 2 * LAG]) * (fps / (2 * LAG))
    idx = np.arange(0, n, STEP)
    xy, sp, vel, dsp, bxy = xy[idx], sp[idx], vel[idx], dsp[idx], bxy[idx]
    T, K = xy.shape[:2]
    ok = np.isfinite(xy[:, :, 0])
    hasv = np.isfinite(vel[:, :, 0])
    feat = np.stack([xy[:, :, 0] / 52.5, xy[:, :, 1] / 34.0,
                     np.nan_to_num(vel[:, :, 0]) / 5, np.nan_to_num(vel[:, :, 1]) / 5,
                     np.nan_to_num(sp) / 5, np.clip(np.nan_to_num(dsp), -8, 8) / 3,
                     np.broadcast_to(np.where(team == 0, 1.0, -1.0), (T, K)),
                     hasv.astype(np.float32)], -1).astype(np.float32)
    feat[~ok] = 0
    order = np.argsort(~ok, axis=1, kind="stable")[:, :SLOTS]  # tracked players first
    X = np.take_along_axis(feat, order[:, :, None], 1)
    M = np.take_along_axis(ok, order, 1)
    if K < SLOTS:
        X = np.pad(X, ((0, 0), (0, SLOTS - K), (0, 0)))
        M = np.pad(M, ((0, 0), (0, SLOTS - K)))
    return X, M, bxy / np.array([52.5, 34.0], np.float32), a + idx


def load_match(path):
    """[(X, mask, y, ball)] per half, y in 0 H, 1 A, 2 D, -1 ungraded; plus what scoring needs."""
    d = np.load(path)
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    lab = d["labels"]
    code = np.full(len(lab), -1, np.int64)
    for i, c in enumerate(CLASSES):
        code[lab == c] = i
    out = []
    for a, b in halves:
        X, M, bxy, fr = half_tensors(d["players"], d["ball"], d["team"], a, b, tuple(d["pitch"]), int(d["fps"]))
        out.append(dict(X=X, M=M, y=code[fr], ball=bxy, frames=fr))
    return dict(name=os.path.basename(path)[:-4], provider=provider(path), halves=out,
                spans=halves, labels=lab)


def load_all(cache=os.path.join(ROOT, "data", "cache", "deep_v1.pkl")):
    import pickle
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    ms = [load_match(p) for p in external_paths()]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(ms, f, protocol=4)
    return ms


def swap_teams(X, y, ball):
    """The same play seen from the other team: half a turn of the pitch, and home <-> away."""
    X = X.copy()
    X[..., :4] *= -1
    X[..., 6] *= -1
    y = np.where(y == 0, 1, np.where(y == 1, 0, y))
    return X, y, -ball


def mirror(X, ball):
    """Mirror across the long axis (y -> -y)."""
    X = X.copy()
    X[..., 1] *= -1
    X[..., 3] *= -1
    ball = ball.copy()
    ball[..., 1] *= -1
    return X, ball
