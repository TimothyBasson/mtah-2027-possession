"""Simulate PR 71's video ball fill (alldata/video) on the outside matches, for a held-out score.

Only gaps that drop_ball knocked out (true ball known) can be filled. Like alldata/video: every
STEP-th frame of a gap is scanned; a scan picks a ball with probability PICK; a pick lands within
2 m of the true ball 70% of the time, 2-5 m 13%, and elsewhere on the pitch 17% (PR 71's 382
check frames at conf 0.3: 57% picked, 70% within 2 m, 83% within 5 m, median 0.9 m). Picks are
then cleaned, joined and low-passed exactly as alldata/video/ballfill.py does.

    VIDEO_FILL=0.57 python external/control_features.py ...   (any value turns it on; it is PICK)
"""
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP, JUMP = 12, 30.0
LINK = 2 * STEP + 1
NEAR, MID = 0.70, 0.13


def _errors(rng, n, pitch):
    """n pick positions as offsets kind: returns (kind, radius) and wrong flags."""
    u = rng.random(n)
    r = np.where(u < NEAR, np.minimum(rng.rayleigh(0.75, n), 2.0), rng.uniform(2, 5, n))
    ang = rng.uniform(0, 2 * np.pi, n)
    off = np.c_[r * np.cos(ang), r * np.sin(ang)]
    wrong = u >= NEAR + MID
    anywhere = np.c_[rng.uniform(0, pitch[0], n), rng.uniform(0, pitch[1], n)]
    return off, wrong, anywhere


def _drop_jumps(fr, xy, fps):
    keep = np.ones(len(fr), bool)
    for i in range(len(fr)):
        bad = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(fr) and abs(fr[j] - fr[i]) <= LINK:
                bad.append(np.linalg.norm(xy[j] - xy[i]) / (abs(fr[j] - fr[i]) / fps) > JUMP)
        keep[i] = not (bad and all(bad))
    return fr[keep], xy[keep]


def fill(true_ball, knocked_ball, fps, pick, pitch=(105.0, 68.0)):
    sys.path.insert(0, os.path.join(ROOT, "alldata", "learned"))
    from feats import lowpass
    truth = true_ball[:, :2].astype(np.float64)
    b = knocked_ball[:, :2].astype(np.float64).copy()
    seen = np.isfinite(b[:, 0])
    can = np.isfinite(truth[:, 0]) & ~seen      # knocked out by drop_ball, truth known
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(truth[:200]).tobytes()) ^ 71)
    filled = np.zeros(len(b), bool)
    d = np.diff(np.r_[0, (~seen).astype(int), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        fr = np.arange(s, e, STEP)
        fr = fr[can[fr] & (rng.random(len(fr)) < pick)]
        if not len(fr):
            continue
        off, wrong, anywhere = _errors(rng, len(fr), pitch)
        xy = np.where(wrong[:, None], anywhere, truth[fr] + off)
        if s - 1 >= 0 and seen[s - 1]:
            fr, xy = np.r_[s - 1, fr], np.r_[b[s - 1:s], xy]
        if e < len(b) and seen[e]:
            fr, xy = np.r_[fr, e], np.r_[xy, b[e:e + 1]]
        fr, xy = _drop_jumps(fr, xy, fps)
        for i in range(len(fr)):
            if not (s <= fr[i] < e):
                continue
            lo, hi = max(fr[i] - STEP // 2, s), min(fr[i] + STEP // 2 + 1, e)
            b[lo:hi] = xy[i]
            filled[lo:hi] = True
        for i in range(len(fr) - 1):
            f0, f1 = fr[i], fr[i + 1]
            if f1 - f0 <= LINK:
                t = np.arange(max(f0, s), min(f1, e))
                if len(t):
                    w = ((t - f0) / (f1 - f0))[:, None]
                    b[t] = xy[i] * (1 - w) + xy[i + 1] * w
                    filled[t] = True
    lp = lowpass(b, 5, 0.7, fps)
    b[filled] = lp[filled]
    out = knocked_ball.copy()
    out[:, :2] = b
    if out.shape[1] > 2:
        sp = np.full(len(b), np.nan)
        sp[5:] = np.linalg.norm(b[5:] - b[:-5], axis=-1) * fps / 5
        out[filled, 2] = sp[filled]
    return out
