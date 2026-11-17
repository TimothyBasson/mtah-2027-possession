"""The ball's path: velocity, velocity change, and the touches in it.

Written for this folder, independent of hmm/ball.py. A ball in flight or rolling changes velocity
slowly; a kick, a trap, a header or a deflection changes it at once. So

    dv(t) = | v(t .. t+k) - v(t-k .. t) |        k = 0.32 s, on a path smoothed as BAR-ATH's is (common.py)

peaks at a touch. A touch is a local peak of dv over TOUCH_DV with a player within TOUCH_RADIUS;
its team is the nearer team's (+1 home, -1 away), or 0 when nobody is that close (a bounce, the
post, a tracker jump). Nothing is measured across a gap in the ball's track.
"""

import numpy as np

K_SECONDS = 0.32
TOUCH_DV = 2.5         # m/s
TOUCH_RADIUS = 3.5     # m
TOUCH_SEPARATION = 0.48  # s between two touches
MAX_BALL_SPEED = 45.0  # m/s: faster is the tracker jumping to another object


def kin(bx, by, fps):
    """Speeds and velocity change per frame from positions with NaN where the ball is not shown."""
    n = len(bx)
    k = max(1, int(round(K_SECONDS * fps)))
    nan = np.full(n, np.nan)
    vbx, vby, vax, vay = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    vbx[k:] = (bx[k:] - bx[:-k]) * fps / k
    vby[k:] = (by[k:] - by[:-k]) * fps / k
    vax[:-k] = vbx[k:]
    vay[:-k] = vby[k:]
    h = max(1, k // 2)
    vx, vy = nan.copy(), nan.copy()
    vx[h:-h] = (bx[2 * h:] - bx[:-2 * h]) * fps / (2 * h)
    vy[h:-h] = (by[2 * h:] - by[:-2 * h]) * fps / (2 * h)
    speed = np.hypot(vx, vy)
    bad = speed > MAX_BALL_SPEED
    speed[bad] = np.nan
    dv = np.hypot(vax - vbx, vay - vby)
    dv[np.hypot(vax, vay) > MAX_BALL_SPEED] = np.nan
    dv[np.hypot(vbx, vby) > MAX_BALL_SPEED] = np.nan
    sb, sa = np.hypot(vbx, vby), np.hypot(vax, vay)
    # one-sided velocity for the edges of a gap: whichever side exists
    vx_edge = np.where(np.isfinite(vx), vx, np.where(np.isfinite(vbx), vbx, vax))
    vy_edge = np.where(np.isfinite(vy), vy, np.where(np.isfinite(vby), vby, vay))
    return dict(vx=vx, vy=vy, speed=speed, dv=dv, acc=sa - sb, speed_before=sb, speed_after=sa,
                vx_edge=vx_edge, vy_edge=vy_edge, speed_edge=np.hypot(vx_edge, vy_edge))


def touches(k, d_home, d_away, fps, dv_min=TOUCH_DV, radius=TOUCH_RADIUS):
    """Touches as arrays: frame, team (+1 home, -1 away, 0 nobody within radius), dv, speed_after,
    margin (how much nearer the touching team's player is than the other's, m)."""
    dv = np.nan_to_num(k["dv"], nan=0.0)
    n = len(dv)
    sep = max(1, int(round(TOUCH_SEPARATION * fps)))
    cand = np.flatnonzero(dv >= dv_min)
    keep = []
    # greedy non-maximum suppression, strongest first
    order = cand[np.argsort(-dv[cand], kind="stable")]
    taken = np.zeros(n, bool)
    for i in order:
        if not taken[i]:
            keep.append(i)
            taken[max(0, i - sep):i + sep + 1] = True
    fr = np.array(sorted(keep), int)
    w = max(1, int(round(0.2 * fps)))
    team = np.zeros(len(fr), int)
    margin = np.zeros(len(fr))
    for j, i in enumerate(fr):
        a, b = max(0, i - w), min(n, i + w + 1)
        h, aw = np.nanmin(d_home[a:b]) if np.isfinite(d_home[a:b]).any() else np.inf, \
            np.nanmin(d_away[a:b]) if np.isfinite(d_away[a:b]).any() else np.inf
        if min(h, aw) <= radius:
            team[j] = 1 if h < aw else -1
            margin[j] = min(abs(h - aw), 10.0)
    return dict(frame=fr, team=team, dv=dv[fr], speed_after=np.nan_to_num(k["speed_after"][fr], nan=0.0),
                margin=margin)
