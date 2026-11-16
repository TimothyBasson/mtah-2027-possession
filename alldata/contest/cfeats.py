"""Contest columns for the all-data model: who the ball came off and who it is going to, read off
the ball the model is shown (BAR-ATH-like: smoothed, with gaps).

    python alldata/contest/cfeats.py [--versions 3] [--jobs 4] [--check]

A smoothed ball rounds every corner, so at a one-touch block or a first-time clearance the shown
ball never reaches the player who played it (it turns 2 to 4 m short), and feats.py's touch
columns, which ask who is nearest the shown ball at the turn, name nobody or the wrong team.
Away from the corner the shown ball is the true one, so the straight stretches before and after
a turn can be run on to where they meet: that point and time are where the ball was really
played. Columns (written to data/cache/alldata/contest/<match>.npz, rows as feats.py's):

  k_*      the last and the next turn of the ball ("kink"): how long ago / ahead, how sharp, how
           well the two lines meet, each team's nearest player to the meeting point, the margin
           between them, and a signed run of kinks won by one team
  h_/a_org the ball's present line run backwards against where the players were: each team's
           nearest pass to it (who played this ball), and h_/a_rcv the same forwards (who it is
           travelling to)
  o_m/r_m  the two margins

--check prints how often the kink names the team of a real touch (found on the ball as recorded)
against naming the team nearest the shown ball, as feats.py's touch columns do.
"""

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LEARNED = os.path.join(ROOT, "alldata", "learned")
sys.path[:0] = [LEARNED, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), ROOT]
import data as D  # noqa: E402
import feats  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "alldata", "contest")
STEP = feats.STEP
KINK_DV = 2.0          # m/s change of velocity over 0.16 s, at its local peak (feats.TOUCH_DV)
ARMS = (0.6, 0.88)     # s either side of a kink where the shown ball is taken to be straight again
CLIP_T = 5.0
NAMES = []


def _build():
    spec = importlib.util.spec_from_file_location("alldata_build", os.path.join(LEARNED, "build.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def maxpool(x, r):
    out = x.copy()
    for s in range(1, r + 1):
        out[s:] = np.maximum(out[s:], x[:-s])
        out[:-s] = np.maximum(out[:-s], x[s:])
    return out


def kinematics(bxy, fps):
    n = len(bxy)
    seen = np.isfinite(bxy[:, 0])
    idx = np.arange(n)
    if seen.any():
        bi = np.stack([np.interp(idx, idx[seen], bxy[seen, k]) for k in (0, 1)], 1)
    else:
        bi = np.zeros((n, 2))
    k = 2
    v = np.zeros_like(bi)
    v[k:-k] = (bi[2 * k:] - bi[:-2 * k]) * (fps / (2 * k))
    v = np.clip(v, -40, 40)
    acc = np.zeros(n)
    acc[k:-k] = np.linalg.norm(v[2 * k:] - v[:-2 * k], axis=1)
    return seen, bi, v, acc


def team_dist(pxy, team, frames, pts):
    """(len(frames), 2): each team's nearest player at frames[i] to pts[i]."""
    p = pxy[frames]
    d = np.linalg.norm(np.nan_to_num(p, nan=1e4) - pts[:, None], axis=-1)
    d[~np.isfinite(p[:, :, 0])] = 1e4
    return np.stack([np.where((team == t)[None], d, 1e4).min(1) for t in (0, 1)], 1)


def kinks(bxy, pxy, team, fps):
    """Turns of the shown ball. Returns frames, the reconstructed frame and point, each team's
    distance to it (K, 2), sharpness (m/s), and how far apart the two lines pass (m)."""
    n = len(bxy)
    seen, bi, v, acc = kinematics(bxy, fps)
    cand = seen & (acc > KINK_DV) & (acc >= maxpool(acc, 6))
    tf = np.flatnonzero(cand)
    if not len(tf):
        z = np.zeros(0)
        return tf, tf, np.zeros((0, 2)), np.zeros((0, 2)), z, z, np.zeros((0, 2))
    cs = np.r_[0, np.cumsum(~seen)]
    best_r = np.full(len(tf), np.inf)
    t_star = tf.astype(float)
    x_star = bi[tf].copy()
    dv = acc[tf].copy()
    for arm in ARMS:
        w = int(round(arm * fps))
        a, b = tf - w, tf + w
        ok = (a >= 2) & (b < n - 2)
        a, b = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
        ok &= (cs[b + 1] - cs[a]) == 0           # the ball is seen all the way across
        d = arm
        vi, vo, pi, po = v[a], v[b], bi[a], bi[b]
        dvv = vi - vo
        den = (dvv ** 2).sum(1)
        ok &= den > 1.0
        u = -(dvv * (pi - po + (vi + vo) * d)).sum(1) / np.maximum(den, 1e-9)
        u = np.clip(u, -d, d)
        xin, xout = pi + vi * (u + d)[:, None], po + vo * (u - d)[:, None]
        r = np.linalg.norm(xin - xout, axis=1)
        take = ok & (r < best_r)
        best_r[take] = r[take]
        t_star[take] = tf[take] + u[take] * fps
        x_star[take] = (xin[take] + xout[take]) / 2
        dv[take] = np.sqrt(den[take])
    fr = np.clip(np.rint(t_star).astype(int), 0, n - 1)
    dk = team_dist(pxy, team, fr, x_star)
    # the plain reading: nearest to the shown ball around the turn (what feats.py's touch columns use)
    dall = [np.where((team == t)[None], np.where(np.isfinite(pxy[:, :, 0]), np.linalg.norm(np.nan_to_num(pxy, nan=1e4) - bi[:, None], axis=-1), 1e4), 1e4).min(1) for t in (0, 1)]
    plain = np.stack([-maxpool(-dall[t], 8)[tf] for t in (0, 1)], 1)
    best_r[~np.isfinite(best_r)] = 9.0
    return tf, fr, x_star, dk, dv, best_r, plain


def rays(bi, v, pxy, team, sel, fps, horizon, sign):
    """Each team's nearest pass to the ball's line through frames sel, run on for up to
    horizon[i] seconds: forwards (sign +1, against where players will be) or backwards (-1)."""
    n = len(bi)
    out = np.full((len(sel), 2), 1e4)
    for tau in np.arange(0.0, 2.6, 0.2):
        f = np.clip(sel + sign * int(round(tau * fps)), 0, n - 1)
        d = team_dist(pxy, team, f, bi[sel] + sign * v[sel] * tau)
        d[tau > horizon + 1e-9] = 1e4
        out = np.minimum(out, d)
    return out


def half_columns(pxy, bxy, team, fps):
    """(T, F) contest columns for one half; pxy (n, K, 2) and bxy (n, 2) in the same metres."""
    n = len(bxy)
    seen, bi, v, acc = kinematics(bxy, fps)
    speed = np.linalg.norm(v, axis=1)
    tf, fr, xs, dk, dv, res, _ = kinks(bxy, pxy, team, fps)
    idx = np.arange(n)
    sel = np.arange(0, n, STEP)
    cols, names = [], []

    def add(name, x):
        cols.append(np.asarray(x, np.float32)); names.append(name)

    K = len(tf)
    if K:
        margin = np.clip(dk[:, 1] - dk[:, 0], -5, 5) / 5           # + home nearer
        won = np.where((dk.min(1) < 2.5) & (np.abs(dk[:, 1] - dk[:, 0]) > 0.5), np.sign(dk[:, 1] - dk[:, 0]), 0.0)
        run = won.copy()
        for j in range(1, K):
            if won[j] != 0 and won[j] == np.sign(run[j - 1]) and tf[j] - tf[j - 1] < 10 * fps:
                run[j] = run[j - 1] + won[j]
        prev = np.searchsorted(tf, sel, side="right") - 1
        nx = prev + 1
        for tag, j, has, dt in (("last", np.clip(prev, 0, K - 1), prev >= 0, None), ("next", np.clip(nx, 0, K - 1), nx < K, None)):
            t = np.abs(sel - tf[j]) / fps
            fresh = has & (t < CLIP_T)
            add(f"k_{tag}_t", np.where(has, np.clip(t, 0, CLIP_T), CLIP_T) / CLIP_T)
            add(f"h_k{tag}", np.where(fresh, np.exp(-dk[j, 0] / 1.5), 0))
            add(f"a_k{tag}", np.where(fresh, np.exp(-dk[j, 1] / 1.5), 0))
            add(f"k_{tag}_m", np.where(fresh, margin[j], 0))
            add(f"k_{tag}_dv", np.where(fresh, np.log1p(dv[j]) / 3, 0))
            add(f"k_{tag}_res", np.where(fresh, np.exp(-res[j] / 2), 0))
            add(f"k_{tag}_run", np.where(fresh, np.clip(run[j], -6, 6) / 3, 0))
        ago = np.where(prev >= 0, (sel - tf[np.clip(prev, 0, K - 1)]) / fps, 9.0)
        ahead = np.where(nx < K, (tf[np.clip(nx, 0, K - 1)] - sel) / fps, 9.0)
    else:
        for tag in ("last", "next"):
            add(f"k_{tag}_t", np.ones(len(sel)))
            for nm in (f"h_k{tag}", f"a_k{tag}", f"k_{tag}_m", f"k_{tag}_dv", f"k_{tag}_res", f"k_{tag}_run"):
                add(nm, np.zeros(len(sel)))
        ago = ahead = np.full(len(sel), 9.0)
    moving = seen[sel] & (speed[sel] > 3.0)
    org = rays(bi, v, pxy, team, sel, fps, np.minimum(ago + 0.6, 2.4), -1)
    rcv = rays(bi, v, pxy, team, sel, fps, np.minimum(ahead + 0.6, 2.4), +1)
    for nm, d in (("org", org), ("rcv", rcv)):
        add(f"h_{nm}", np.where(moving, np.exp(-d[:, 0] / 2), 0))
        add(f"a_{nm}", np.where(moving, np.exp(-d[:, 1] / 2), 0))
        add(f"{nm[0]}_m", np.where(moving, np.clip(np.minimum(d[:, 1], 20) - np.minimum(d[:, 0], 20), -5, 5) / 5, 0))
    NAMES[:] = names
    return np.stack(cols, 1).astype(np.float32)


SIGNED = ("k_last_m", "k_next_m", "k_last_run", "k_next_run", "o_m", "r_m")


def symmetry(names):
    """feats.symmetry plus the signed contest columns (the h_/a_ pairs are handled there)."""
    perm, sw, mi = feats.symmetry(names)
    for i, nm in enumerate(names):
        if nm in SIGNED:
            sw[i] = -1
    return perm, sw, mi


def one(args):
    path, versions = args
    name = os.path.basename(path)[:-4]
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        return name
    build = _build()
    d = dict(np.load(path))
    fps = int(d["fps"])
    pxy = d["players"][:, :, :2].astype(np.float64)
    out = {}
    for v in range(versions + 1):
        bxy = build.ball_version(d, name, v).astype(np.float64)
        for i, (a, b) in enumerate((int(h[0]), int(h[1])) for h in d["halves"]):
            out[f"F{v}_{i}"] = half_columns(pxy[a:b + 1], bxy[a:b + 1], d["team"], fps)
    os.makedirs(OUT, exist_ok=True)
    np.savez(dst + ".tmp.npz", names=np.array(NAMES), **out)
    os.replace(dst + ".tmp.npz", dst)
    return name


def check(path):
    """Team of real touches (recorded ball: velocity jump with a player within 1.5 m, the other
    team at least 1 m further) against the kink's reading and the plain reading of the shown ball."""
    build = _build()
    name = os.path.basename(path)[:-4]
    d = dict(np.load(path))
    fps = int(d["fps"])
    pxy, team = d["players"][:, :, :2].astype(np.float64), d["team"]
    raw = feats.hide_frozen(d["ball"][:, :2].astype(np.float64), fps)
    sim = build.ball_version(d, name, 1).astype(np.float64)
    seen, bi, v, acc = kinematics(raw, fps)
    tr = np.flatnonzero(seen & (acc > 4.0) & (acc >= maxpool(acc, 6)))
    dr = team_dist(pxy, team, tr, bi[tr])
    ok = (dr.min(1) < 1.5) & (np.abs(dr[:, 0] - dr[:, 1]) > 1.0)
    tr, true = tr[ok], dr[ok].argmin(1)
    tf, fr, xs, dk, dv, res, plain = kinks(sim, pxy, team, fps)
    if not len(tf) or not len(tr):
        return name, np.zeros(6)
    j = np.clip(np.searchsorted(tf, tr), 1, len(tf) - 1)
    j = np.where(np.abs(tf[j - 1] - tr) < np.abs(tf[j] - tr), j - 1, j)
    hit = np.abs(tf[j] - tr) <= 0.5 * fps
    contested = dr.max(1)[ok.nonzero()[0] * 0 + np.arange(len(tr))] if False else None
    both = np.sort(dr[ok], 1)[:, 1] < 4.0                      # an opponent within 4 m of the touch
    r = []
    for m in (hit, hit & both):
        r += [m.sum(), (dk[j[m]].argmin(1) == true[m]).sum(), (plain[j[m]].argmin(1) == true[m]).sum()]
    return name, np.array(r, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    paths = sorted(p for p in D.external_paths() if "soccertrack" not in p)
    with ProcessPoolExecutor(a.jobs) as ex:
        if a.check:
            R = {}
            for name, r in ex.map(check, paths):
                R.setdefault(name.split("_")[0], []).append(r)
            print("real touches with a kink within 0.5 s: team named right by the kink / by the nearest player to the shown ball")
            for pv, rs in R.items():
                s = np.sum(rs, 0)
                print(f"  {pv:12s} all {int(s[0]):6d}  {s[1] / s[0]:.3f} / {s[2] / s[0]:.3f}   opponent within 4 m {int(s[3]):6d}  {s[4] / s[3]:.3f} / {s[5] / s[3]:.3f}")
            return
        for r in ex.map(one, [(p, a.versions) for p in paths]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
