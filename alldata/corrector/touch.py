"""Touches read off a smoothed ball, with the place of each touch put back where it happened.

A tracker that smooths the ball (BAR-ATH's is a 0.7 Hz low-pass) turns every kick into a round
bend about a second long, and the bend cuts the corner: at the moment of a hard touch the
smoothed ball is 1 to 4 m from where the ball really was, which is enough to hand the touch to
the wrong player when two are close. Between touches a ball moves in a nearly straight line, so
the true path is piecewise linear and the smoothed one is that path through a symmetric filter K.
Then the smoothed ball's acceleration is a sum of copies of K, one per touch, each scaled by
that touch's change of velocity dv, and for a touch at time t

    true position(t) = smoothed position(t) - 1/2 * integral |s| * acceleration(t + s) ds

whatever K is (the integral is dv * int_0^inf s K(s) ds). No filter has to be known or fitted.
"""

import numpy as np


def _runs(mask):
    e = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    s, t = np.r_[0, e], np.r_[e, len(mask)]
    return s, t, mask[s]


def kinematics(bxy, fps=25):
    """Velocity and acceleration (n, 2) of the ball by central differences; NaN beside gaps."""
    v = np.full_like(bxy, np.nan, dtype=np.float64)
    a = np.full_like(bxy, np.nan, dtype=np.float64)
    v[1:-1] = (bxy[2:] - bxy[:-2]) * (fps / 2)
    a[1:-1] = (bxy[2:] - 2 * bxy[1:-1] + bxy[:-2]) * fps * fps
    return v, a


def touches(bxy, fps=25, min_dv=1.5, win=0.8, sep=0.32):
    """Touch events of a ball path (n, 2), NaN where unseen.

    Returns a dict of arrays, one entry per touch: frame, dv (2,), pos (2,) the corrected place,
    obs (2,) the smoothed ball there, vin / vout (2,) velocity before and after.
    win: half-length in seconds of the stretch a bend is integrated over; sep: closest two
    touches may be."""
    n = len(bxy)
    b = bxy.astype(np.float64)
    v, a = kinematics(b, fps)
    W, S = int(round(win * fps)), int(round(sep * fps))
    # light smoothing of the acceleration so raw (unsmoothed) balls do not fire on noise
    k = np.ones(5) / 5
    am = np.hypot(np.convolve(np.nan_to_num(a[:, 0]), k, "same"), np.convolve(np.nan_to_num(a[:, 1]), k, "same"))
    am[~np.isfinite(a[:, 0])] = 0
    peak = am.copy()
    for s in range(1, S + 1):
        peak[s:] = np.maximum(peak[s:], am[:-s])
        peak[:-s] = np.maximum(peak[:-s], am[s:])
    cand = np.flatnonzero((am >= peak) & (am > min_dv * 0.6))
    fr, dv, pos, obs, vin, vout = [], [], [], [], [], []
    last = -10 ** 9
    for t in cand:
        if t - last < S:
            continue
        # the bend's stretch: up to W either side, cut at a gap and half-way to neighbouring peaks
        lo, hi = max(t - W, 1), min(t + W, n - 2)
        seg = np.isfinite(a[lo:hi + 1, 0])
        if not seg[t - lo]:
            continue
        bad = np.flatnonzero(~seg)
        if len(bad):
            lo2 = lo + bad[bad < t - lo].max() + 1 if (bad < t - lo).any() else lo
            hi2 = lo + bad[bad > t - lo].min() - 1 if (bad > t - lo).any() else hi
            lo, hi = lo2, hi2
        s = (np.arange(lo, hi + 1) - t) / fps
        aa = a[lo:hi + 1]
        d = aa.sum(0) / fps
        if np.hypot(*d) < min_dv:
            continue
        off = 0.5 * (np.abs(s)[:, None] * aa).sum(0) / fps
        fr.append(t); dv.append(d); obs.append(b[t]); pos.append(b[t] - off)
        vin.append(v[lo]); vout.append(v[hi])
        last = t
    arr = lambda x: np.array(x, np.float64).reshape(-1, 2)
    return dict(frame=np.array(fr, int), dv=arr(dv), pos=arr(pos), obs=arr(obs), vin=arr(vin), vout=arr(vout))
