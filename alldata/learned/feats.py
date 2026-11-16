"""Tracking with the ball -> per-frame feature rows for the all-data model (5 fps).

Everything here may read the ball. One row per STEP-th frame of a half, in the canonical frame of
deep/data.py (centre spot at 0, home defending x < 0), built so that swapping the teams and
mirroring the pitch are a column permutation and sign flips (SWAP_PERM, SWAP_SIGN, MIRROR_SIGN).

Blocks: the ball (seen, how long unseen, position carried across gaps, velocity, change of
velocity, where it is against the lines), each team against the ball (nearest players, closing
speed, players near it), each team alone (players tracked, centroid), and optional per-frame
probabilities of other models (the players-only net bt; in the "with Tim" variant, his HMM).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.append(os.path.join(ROOT, "deep"))
import data as D  # noqa: E402

STEP = D.STEP
CLIP_T = 10.0  # s
TOUCH_DV = 2.0      # m/s change of the (smoothed) ball's velocity over 0.16 s, at its local peak
TOUCH_RADIUS = 3.0  # m, nearest player within 0.3 s of the peak


def _runs(mask):
    e = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    s, t = np.r_[0, e], np.r_[e, len(mask)]
    return s, t, mask[s]


def smooth(xy, half_width):
    """BAR-ATH's ball is a smoothed path: a kick takes about a second to reach full speed (its
    acceleration tops out near 30 m/s2, a raw outside ball's at 250). A parabolic window of
    14 to 18 frames' half-width over a raw ball reproduces its statistics (found by the
    alldata/physics thread, checked here), so outside balls are smoothed like this before the
    features are computed. NaN stays NaN; the window is renormalised beside a gap."""
    h = int(np.ceil(half_width))
    t = np.arange(-h, h + 1)
    k = np.clip(1 - (t / half_width) ** 2, 0, None)
    ok = np.isfinite(xy[:, 0])
    w = np.convolve(ok.astype(float), k, "same")
    out = xy.copy()
    for c in (0, 1):
        out[:, c] = np.convolve(np.where(ok, xy[:, c], 0.0), k, "same") / np.maximum(w, 1e-9)
    out[~ok] = np.nan
    return out


def hide_frozen(xy, fps=25, seconds=1.0):
    """IDSSE parks a dead ball: the recorded position does not change at all until play restarts.
    Such runs are treated as unseen, like the Metrica samples' stoppages."""
    xy = xy.copy()
    same = np.r_[False, (xy[1:] == xy[:-1]).all(1)]
    s, t, v = _runs(same)
    for a, b in zip(s[v], t[v]):
        if b - a >= seconds * fps:
            xy[a:b] = np.nan
    return xy


def bar_like(xy, labels, rng, fps=25):
    """An outside match's ball made to look like BAR-ATH's: parked balls hidden, visibility as
    simulate_visibility, then smoothed."""
    xy = simulate_visibility(hide_frozen(xy, fps), labels, rng, fps)
    return smooth(xy, rng.uniform(14.0, 18.0) * fps / 25).astype(np.float32)


def simulate_visibility(xy, labels, rng, fps=25):
    """A ball seen the way BAR-ATH's is: tracked through live play, lost for the core of most
    stoppages and now and then in open play. xy (n, 2) with NaN where unseen; labels H/A/D/X.
    Sources differ (IDSSE sees the ball always, Metrica's samples never in a stoppage), so both
    are pushed to the same randomised pattern: after a stoppage starts the ball stays for u1
    seconds, before it ends it is back for u2 seconds; hidden between. Where the source has no
    ball in a stoppage, it is held still at the last / next seen place for those seconds."""
    xy = xy.copy()
    n = len(xy)
    seen = np.isfinite(xy[:, 0])
    q = rng.uniform(0.6, 1.0)   # BAR-ATH: 59% of frames seen with about 45% dead, so most stoppages are mostly hidden
    m1, m2 = rng.uniform(0.3, 4.0, 2)
    dead = labels == "D"
    s, t, v = _runs(dead)
    reach = 3 * fps  # the source may lose the ball a little before the label says dead
    for a, b in zip(s[v], t[v]):
        pa = np.flatnonzero(seen[max(a - reach, 0):a])
        pb = np.flatnonzero(seen[b:b + reach])
        before = xy[max(a - reach, 0) + pa[-1]] if len(pa) else None
        after = xy[b + pb[0]] if len(pb) else None
        if rng.random() > q:  # this stoppage keeps its ball throughout
            if before is not None and after is not None:
                k = np.flatnonzero(~seen[a:b]) + a
                w = ((k - a + 1) / (b - a + 1))[:, None]
                xy[k] = before * (1 - w) + after * w
            continue
        u1, u2 = int(rng.exponential(m1) * fps), int(rng.exponential(m2) * fps)
        lo, hi = a + u1, b - u2
        if before is not None:
            k = np.flatnonzero(~seen[a:min(lo, b)]) + a
            xy[k] = before
        if after is not None:
            k = np.flatnonzero(~seen[max(hi, a):b]) + max(hi, a)
            xy[k] = after
        if lo < hi:
            xy[lo:hi] = np.nan
    rate = rng.uniform(0, 0.5) / 60 / fps  # gaps per live frame
    live = np.flatnonzero((labels == "H") | (labels == "A"))
    for a in live[rng.random(len(live)) < rate]:
        xy[a:a + int(rng.uniform(1, 20) * fps)] = np.nan
    return xy


NAMES = []


def half_features(players, ball_xy, team, a, b, pitch, fps=25, extra=None):
    """(T, F) float32 for frames a..b (inclusive), sampled every STEP frames. ball_xy (n, 2) in
    metres from the corner, NaN where unseen (callers may have changed what is seen). extra:
    list of (name, (T, 3) H/A/D probabilities) appended as inputs."""
    L, W = float(pitch[0]), float(pitch[1])
    hl, hw = L / 2, W / 2
    xy = players[a:b + 1, :, :2].astype(np.float32) - np.array([hl, hw], np.float32)
    bxy = ball_xy[a:b + 1].astype(np.float32) - np.array([hl, hw], np.float32)
    if not D.home_defends_left(players, team, a, b):
        xy, bxy = -xy, -bxy
    n = len(xy)
    seen = np.isfinite(bxy[:, 0])
    idx = np.arange(n)
    if seen.any():
        si = idx[seen]
        bi = np.stack([np.interp(idx, si, bxy[seen, k]) for k in (0, 1)], 1).astype(np.float32)
        last = np.maximum.accumulate(np.where(seen, idx, -10 ** 6))
        nxt = np.minimum.accumulate(np.where(seen, idx, 10 ** 7)[::-1])[::-1]
    else:
        bi = np.zeros((n, 2), np.float32)
        last, nxt = np.full(n, -10 ** 6), np.full(n, 10 ** 7)
    since = np.clip((idx - last) / fps, 0, CLIP_T) / CLIP_T
    until = np.clip((nxt - idx) / fps, 0, CLIP_T) / CLIP_T
    gap = np.where(seen, 0, np.clip((nxt - last) / fps, 0, 60) / 60)
    k = 2
    v = np.zeros_like(bi)
    v[k:-k] = (bi[2 * k:] - bi[:-2 * k]) * (fps / (2 * k))
    v = np.clip(v, -40, 40)
    acc = np.zeros(n, np.float32)
    acc[k:-k] = np.linalg.norm(v[2 * k:] - v[:-2 * k], axis=1)
    speed = np.linalg.norm(v, axis=1)
    # the ball against the lines (metres inside; negative is out)
    in_x, in_y = hl - np.abs(bi[:, 0]), hw - np.abs(bi[:, 1])
    # players against the ball
    ok = np.isfinite(xy[:, :, 0])
    dist = np.linalg.norm(np.nan_to_num(xy, nan=1e4) - bi[:, None], axis=-1)
    dist[~ok] = 1e4
    psp = np.nan_to_num(players[a:b + 1, :, 2].astype(np.float32))
    psp[psp > D.MAX_SPEED] = 0

    def maxpool(x, r=2):  # max over +-r frames
        out = x.copy()
        for s in range(1, r + 1):
            out[s:] = np.maximum(out[s:], x[:-s])
            out[:-s] = np.maximum(out[:-s], x[s:])
        return out

    sel = np.arange(0, n, STEP)
    cols, names = [], []

    def add(name, x):
        cols.append(np.asarray(x, np.float32)[sel] if len(x) == n else np.asarray(x, np.float32))
        names.append(name)

    add("seen", seen)
    add("since", since)
    add("until", until)
    add("gap", gap)
    add("speed", np.log1p(speed) / 3)
    add("acc", np.log1p(maxpool(acc)) / 3)
    add("in_x", np.clip(in_x, -5, 20) / 10)
    add("in_y", np.clip(in_y, -5, 20) / 10)
    add("goal_y", np.clip(np.abs(bi[:, 1]), 0, 20) / 10)  # distance from the goal's axis
    add("pspeed", np.where(ok, psp, np.nan).mean(1) if False else (psp * ok).sum(1) / np.maximum(ok.sum(1), 1) / 3)
    add("bx", bi[:, 0] / 52.5)
    add("vx", v[:, 0] / 10)
    add("by", bi[:, 1] / 34.0)
    add("vy", v[:, 1] / 10)
    team_cols = {}
    for ti, tn in ((0, "h"), (1, "a")):
        m = team == ti
        dt = dist[:, m]
        part = np.sort(dt, axis=1)[:, :3] if dt.shape[1] >= 3 else np.pad(np.sort(dt, axis=1), ((0, 0), (0, 3 - dt.shape[1])), constant_values=1e4)
        d1, d2, d3 = part[:, 0], part[:, 1], part[:, 2]
        near = dt.argmin(1)
        nsp = psp[:, m][idx, near]
        nxy = np.nan_to_num(xy[:, m][idx, near])
        nv = np.zeros_like(nxy)
        nv[k:-k] = (nxy[2 * k:] - nxy[:-2 * k]) * (fps / (2 * k))
        nv[np.linalg.norm(nv, axis=1) > D.MAX_SPEED] = 0
        rel = bi - nxy
        d1c = np.minimum(d1, 60)
        close = np.zeros(n, np.float32)
        close[k:-k] = -(d1c[2 * k:] - d1c[:-2 * k]) * (fps / (2 * k))
        okt = ok[:, m]
        cx = np.where(okt, xy[:, m, 0], 0).sum(1) / np.maximum(okt.sum(1), 1)
        cvx = np.zeros(n, np.float32)
        cvx[12:-12] = (cx[24:] - cx[:-24]) * (fps / 24)
        f = dict(d1=np.log1p(np.minimum(d1, 60)) / 2, d2=np.log1p(np.minimum(d2, 60)) / 2,
                 d3=np.log1p(np.minimum(d3, 60)) / 2,
                 dmin=np.log1p(np.minimum(-maxpool(-d1), 60)) / 2,
                 touch=np.exp(-d1 / 1.5), nsp=nsp / 5, close=np.clip(close, -15, 15) / 5,
                 relv=np.clip(np.linalg.norm(v - nv, axis=1), 0, 30) / 10,
                 n2=(dt < 2).sum(1) / 3, n5=(dt < 5).sum(1) / 5, cnt=okt.sum(1) / 11,
                 cx=cx / 52.5 * (1 if ti == 0 else 1), cvx=np.clip(cvx, -8, 8) / 3,
                 relx=np.clip(rel[:, 0], -20, 20) / 5)
        team_cols[tn] = f
    tnames = list(team_cols["h"])
    for tn in "ha":
        for nm in tnames:
            add(f"{tn}_{nm}", team_cols[tn][nm])
    # touches: the ball's velocity jumps while a player is beside it. Who touched last and who
    # touches next (the whole match is on the table, so the future is fair game), how long ago,
    # how many in a row by one team.
    dall = np.minimum(np.where((team == 0)[None], dist, 1e4).min(1), 1e4), np.where((team == 1)[None], dist, 1e4).min(1)
    near = -maxpool(-np.minimum(dall[0], dall[1]), 8)
    cand = seen & (acc > TOUCH_DV) & (near < TOUCH_RADIUS) & (acc >= maxpool(acc, 6))
    tf = np.flatnonzero(cand)
    tt = np.where(-maxpool(-dall[0], 8)[tf] <= -maxpool(-dall[1], 8)[tf], 0, 1)
    sgn = np.where(tt == 0, 1.0, -1.0)
    run = np.ones(len(tf))
    for j in range(1, len(tf)):
        if tt[j] == tt[j - 1] and tf[j] - tf[j - 1] < 10 * fps:
            run[j] = run[j - 1] + 1
    prev = np.searchsorted(tf, idx, side="right") - 1   # last touch at or before each frame
    nx = prev + 1
    hp, hn = prev >= 0, nx < len(tf)
    pc, nc = np.clip(prev, 0, max(len(tf) - 1, 0)), np.clip(nx, 0, max(len(tf) - 1, 0))
    if len(tf):
        ago = np.where(hp, (idx - tf[pc]) / fps, CLIP_T)
        ahead = np.where(hn, (tf[nc] - idx) / fps, CLIP_T)
        fresh_p, fresh_n = hp & (ago < CLIP_T), hn & (ahead < CLIP_T)
        add("t_ago", np.clip(ago, 0, CLIP_T) / CLIP_T)
        add("t_ahead", np.clip(ahead, 0, CLIP_T) / CLIP_T)
        add("t_last", np.where(fresh_p, sgn[pc], 0))
        add("t_next", np.where(fresh_n, sgn[nc], 0))
        add("t_run", np.where(fresh_p, np.clip(run[pc], 0, 6) * sgn[pc], 0) / 3)
        add("t_runnext", np.where(fresh_n, np.clip(run[nc], 0, 6) * sgn[nc], 0) / 3)
        cs = np.r_[0, np.cumsum(np.bincount(tf, weights=sgn, minlength=n))]
        w = 5 * fps
        add("t_past5", (cs[idx + 1] - cs[np.maximum(idx + 1 - w, 0)]) / 4)
        add("t_fut5", (cs[np.minimum(idx + w, n)] - cs[idx + 1]) / 4)
    else:
        for nm in ("t_ago", "t_ahead"):
            add(nm, np.ones(n))
        for nm in ("t_last", "t_next", "t_run", "t_runnext", "t_past5", "t_fut5"):
            add(nm, np.zeros(n))
    add("margin", np.clip((np.minimum(dist[:, team == 1].min(1), 60) - np.minimum(dist[:, team == 0].min(1), 60)), -10, 10) / 5)
    T = len(sel)
    for nm, p in (extra or []):
        p = np.asarray(p, np.float32)[:T]
        if len(p) < T:
            p = np.pad(p, ((0, T - len(p)), (0, 0)), mode="edge")
        lp = np.log(np.clip(p, 1e-4, 1)) / 4 + 1
        for j, c in enumerate("HAD"):
            cols.append(lp[:, j]); names.append(f"{nm}_{c}")
    NAMES[:] = names
    return np.stack(cols, 1).astype(np.float32), a + sel


def symmetry(names):
    """Column permutation and signs for the team swap, and signs for the mirror."""
    pos = {n: i for i, n in enumerate(names)}
    perm = np.arange(len(names))
    sw = np.ones(len(names), np.float32)
    mi = np.ones(len(names), np.float32)
    for n, i in pos.items():
        if n in ("bx", "vx", "by", "vy", "margin", "t_last", "t_next", "t_run", "t_runnext", "t_past5", "t_fut5"):
            sw[i] = -1
        if n in ("by", "vy"):
            mi[i] = -1
        if n[:2] in ("h_", "a_"):
            o = ("a_" if n[0] == "h" else "h_") + n[2:]
            perm[i] = pos[o]
            if n[2:] in ("cx", "cvx", "relx"):
                sw[i] = -1
        if n.endswith("_H") and n[:-2] + "_A" in pos:
            perm[i] = pos[n[:-2] + "_A"]
        if n.endswith("_A") and n[:-2] + "_H" in pos:
            perm[i] = pos[n[:-2] + "_H"]
    return perm, sw, mi
