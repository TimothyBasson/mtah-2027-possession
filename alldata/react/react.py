"""Team-reaction features: how each whole team moves just before and just after every moment.

    python alldata/react/react.py [--jobs 4]      # data/cache/alldata/react/<match>.npz

When a team wins the ball its shape opens and it moves up the pitch, while the other team turns,
drops and closes up. These rows describe exactly that, per team, at 5 fps, in windows that look
back (2 s, 1 s) and ahead (1 s, 2 s, 4 s): the match is labelled offline, so the future is there.

Per team and window, over the players tracked at both ends of the window (so a player appearing
or vanishing is not read as movement), in the team's own direction of attack:
  adv    mean advance along the pitch (m/s)
  spr    change of the mean distance from the team's centroid (opening / closing)
  len    change of the team's length (spread along the pitch)
  wid    change of its width
  fwd    share of players advancing faster than 0.5 m/s
  back   advance of the back four (2nd to 5th deepest: the deepest is the keeper)
  front  advance of the front three
  toball change of the mean distance to the ball (closing on it or moving away), 0 when unseen
Per team at the moment itself: centroid, spread, length, width, mean speed, players behind the
ball. And both teams together: the sum of the two advances in pitch coordinates (both teams
moving the same way is the reaction; opposite ways is not).

The ball used for `toball` and `behind` is the BAR-ATH-like ball of build.ball_version 1 (0 for
BAR-ATH itself), the one the base model saw.
"""

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.append(os.path.join(ROOT, "deep"))
import data as D  # noqa: E402

STEP = D.STEP
OUT = os.path.join(ROOT, "data", "cache", "alldata", "react")
WINDOWS = (("p2", -10), ("p1", -5), ("f1", 5), ("f2", 10), ("f4", 20))  # steps of 0.2 s
PER_WINDOW = ("adv", "spr", "len", "wid", "fwd", "back", "front", "toball")
STATIC = ("cu", "spread", "length", "width", "speed", "behind", "count")
NAMES = ([f"{t}_{w}_{f}" for t in "ha" for w, _ in WINDOWS for f in PER_WINDOW] +
         [f"{t}_{s}" for t in "ha" for s in STATIC] + [f"both_{w}" for w, _ in WINDOWS] + ["ball_seen"])


def _pair(x, k):
    """x at the start and the end of the window that runs k steps from each row (k < 0: the
    window ends at the row). Rows whose window leaves the half get NaN."""
    T = len(x)
    a = np.full_like(x, np.nan)
    b = np.full_like(x, np.nan)
    if k > 0:
        a[:T - k], b[:T - k] = x[:T - k], x[k:]
    else:
        a[-k:], b[-k:] = x[:T + k], x[-k:]
    return a, b


def _nanmean(x, axis=1):
    ok = np.isfinite(x)
    return np.where(ok, x, 0).sum(axis) / np.maximum(ok.sum(axis), 1)


def _std(x, m):
    n = np.maximum(m.sum(1), 1)
    mu = np.where(m, x, 0).sum(1) / n
    return np.sqrt(np.where(m, (x - mu[:, None]) ** 2, 0).sum(1) / n)


def _shape(u, y, m):
    n = np.maximum(m.sum(1), 1)
    cu, cy = np.where(m, u, 0).sum(1) / n, np.where(m, y, 0).sum(1) / n
    r = np.hypot(u - cu[:, None], y - cy[:, None])
    return cu, np.where(m, r, 0).sum(1) / n, _std(u, m), _std(y, m)


def half_rows(players, ball_xy, team, a, b, pitch, fps=25):
    """(T, len(NAMES)) float32 at STEP spacing for frames a..b (inclusive)."""
    L, W = float(pitch[0]), float(pitch[1])
    xy = players[a:b + 1, :, :2].astype(np.float32) - np.array([L / 2, W / 2], np.float32)
    bxy = ball_xy[a:b + 1].astype(np.float32) - np.array([L / 2, W / 2], np.float32)
    if not D.home_defends_left(players, team, a, b):
        xy, bxy = -xy, -bxy
    n = len(xy)
    sel = np.arange(0, n, STEP)
    # positions averaged over the STEP frames around each row (trackers jitter)
    pad = np.pad(xy, ((STEP // 2, STEP // 2), (0, 0), (0, 0)), constant_values=np.nan)
    st = np.stack([pad[i:i + n][sel] for i in range(STEP)])
    okc = np.isfinite(st).sum(0)
    xy5 = np.where(okc > 0, np.nansum(st, 0) / np.maximum(okc, 1), np.nan)
    seen = np.isfinite(bxy[:, 0])
    idx = np.arange(n)
    if seen.any():
        bi = np.stack([np.interp(idx, idx[seen], bxy[seen, c]) for c in (0, 1)], 1)[sel]
    else:
        bi = np.zeros((len(sel), 2))
    bs = seen[sel]
    psp = players[a:b + 1, :, 2].astype(np.float32)[sel]
    psp = np.where(psp > D.MAX_SPEED, np.nan, psp)
    T = len(sel)
    cols = {}
    adv_pitch = {}
    for ti, tn, sgn in ((0, "h", 1.0), (1, "a", -1.0)):
        m = team == ti
        u, y = xy5[:, m, 0] * sgn, xy5[:, m, 1]
        bu = bi[:, 0] * sgn
        ok = np.isfinite(u)
        dball = np.hypot(u - bu[:, None], y - bi[:, 1][:, None])
        for wn, k in WINDOWS:
            tau = abs(k) * STEP / fps
            u0, u1 = _pair(u, k)
            y0, y1 = _pair(y, k)
            d0, d1 = _pair(dball, k)
            s0, s1 = _pair(bs[:, None].astype(np.float32), k)
            both = np.isfinite(u0) & np.isfinite(u1)
            cnt = np.maximum(both.sum(1), 1)
            du = np.where(both, u1 - u0, 0)
            adv = du.sum(1) / cnt / tau
            sh0, sh1 = _shape(u0, y0, both), _shape(u1, y1, both)
            # back four and front three by depth at the window's start
            order = np.argsort(np.where(both, u0, np.inf), axis=1)
            dsort = np.take_along_axis(du, order, 1)
            nb = both.sum(1)
            back = dsort[:, 1:5].sum(1) / np.clip(nb - 1, 1, 4) / tau
            rank = np.arange(u.shape[1])[None]
            fr = np.where((rank >= (nb - 3)[:, None]) & (rank < nb[:, None]), dsort, 0).sum(1) / np.clip(nb, 1, 3) / tau
            tb = np.where(both, d1 - d0, 0).sum(1) / cnt / tau * (s0[:, 0] * s1[:, 0] > 0)
            f = dict(adv=adv, spr=(sh1[1] - sh0[1]) / tau, len=(sh1[2] - sh0[2]) / tau, wid=(sh1[3] - sh0[3]) / tau,
                     fwd=np.where(both, du / tau > 0.5, 0).sum(1) / cnt, back=back, front=fr, toball=np.nan_to_num(tb))
            for nm in PER_WINDOW:
                cols[f"{tn}_{wn}_{nm}"] = np.where(both.any(1), f[nm], 0)
            adv_pitch[(tn, wn)] = np.where(both.any(1), adv * sgn, 0)
        cu, spread, length, width = _shape(u, y, ok)
        cols[f"{tn}_cu"], cols[f"{tn}_spread"], cols[f"{tn}_length"], cols[f"{tn}_width"] = cu / 52.5, spread / 20, length / 20, width / 20
        cols[f"{tn}_speed"] = _nanmean(psp[:, m]) / 3
        cols[f"{tn}_behind"] = np.where(bs, (ok & (u < bu[:, None])).sum(1) / np.maximum(ok.sum(1), 1), 0.5)
        cols[f"{tn}_count"] = ok.sum(1) / 11
    for wn, _ in WINDOWS:
        cols[f"both_{wn}"] = adv_pitch[("h", wn)] + adv_pitch[("a", wn)]
    cols["ball_seen"] = bs.astype(np.float32)
    F = np.stack([np.nan_to_num(np.asarray(cols[nm], np.float32)) for nm in NAMES], 1)
    assert F.shape == (T, len(NAMES))
    return F, a + sel


def swap(F):
    """The same rows with the teams exchanged (labels H and A exchange too)."""
    pos = {n: i for i, n in enumerate(NAMES)}
    perm = np.arange(len(NAMES))
    sign = np.ones(len(NAMES), np.float32)
    for n, i in pos.items():
        if n[:2] in ("h_", "a_"):
            perm[i] = pos[("a_" if n[0] == "h" else "h_") + n[2:]]
        if n.startswith("both_"):
            sign[i] = -1
    return F[:, perm] * sign


def load_build():
    spec = importlib.util.spec_from_file_location("alldata_build", os.path.join(ROOT, "alldata", "learned", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one(path):
    name = os.path.basename(path)[:-4]
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        return name
    build = load_build()
    d = dict(np.load(path))
    bxy = build.ball_version(d, name, 1)
    out = {}
    for i, (a, b) in enumerate((int(h[0]), int(h[1])) for h in d["halves"]):
        out[f"R{i}"], out[f"fr{i}"] = half_rows(d["players"], bxy, d["team"], a, b, d["pitch"], int(d["fps"]))
    os.makedirs(OUT, exist_ok=True)
    np.savez(dst + ".tmp.npz", **out)
    os.replace(dst + ".tmp.npz", dst)
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    paths = sorted(p for p in D.external_paths() if "soccertrack" not in p)
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, paths):
            print(r, flush=True)
