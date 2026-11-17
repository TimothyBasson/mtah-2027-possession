"""Pick one ball per scanned frame from the detector's boxes and turn the picks into a ball track.

Filter (pitch geometry only, nothing tied to frames of this match): the box must project inside the
pitch lines (spare balls sit by the touchlines), and not within SPOT m of a penalty spot or the
centre spot (painted spots read as balls). Of what is left, the most confident box with
conf >= MIN_CONF is the ball."""
import glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "possession")]
import load, pitch

SPOT, MIN_CONF, EDGE = 1.0, 0.3, 0.0


def detections():
    return np.concatenate([np.load(f)["det"] for f in sorted(glob.glob(os.path.join(ROOT, "data/cache/video/det_*.npz")))])


def scanned_frames():
    """Frames detect.py actually decoded: every decoded frame has at least one row (a NaN row when
    nothing was found), so the rows are the exact set."""
    return np.unique(detections()[:, 0]).astype(int)


def picks(min_conf=MIN_CONF):
    """{frame: (x, y metres, conf)} for the scanned frames where a ball is picked."""
    L, W = load.PITCH
    det = detections()
    det = det[~np.isnan(det[:, 1]) & (det[:, 5] >= min_conf)]
    fr = det[:, 0].astype(int)
    H, _ = pitch.homographies()
    xy = pitch.to_pitch(H, fr, det[:, 1:3]) * np.array(load.PITCH)
    spots = np.array([[11.0, W / 2], [L - 11.0, W / 2], [L / 2, W / 2]])
    ok = (xy[:, 0] >= -EDGE) & (xy[:, 0] <= L + EDGE) & (xy[:, 1] >= -EDGE) & (xy[:, 1] <= W + EDGE)
    ok &= np.min(np.linalg.norm(xy[:, None] - spots[None], axis=2), axis=1) > SPOT
    best = {}
    for f, p, c in zip(fr[ok], xy[ok], det[ok, 5]):
        if f not in best or c > best[f][2]:
            best[f] = (p[0], p[1], c)
    return best


STEP = 12          # targets.py scans every 12th gap frame
LINK = 2 * STEP + 1  # neighbouring anchors this close are joined by a straight line
JUMP = 30.0        # m/s: a pick this fast from both of its neighbours is dropped


def _drop_jumps(fr, xy):
    keep = np.ones(len(fr), bool)
    for i in range(len(fr)):
        bad = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(fr) and abs(fr[j] - fr[i]) <= LINK:
                bad.append(np.linalg.norm(xy[j] - xy[i]) / (abs(fr[j] - fr[i]) / load.FPS) > JUMP)
        keep[i] = not (bad and all(bad))
    return fr[keep], xy[keep]


def fill(ball, halves, max_gap=None, min_conf=MIN_CONF, smooth=True):
    """The tracker's ball (n, 3) with its gaps filled from the video picks where they reach.
    max_gap: only gaps up to this many frames. Returns (ball, filled mask)."""
    import feats  # alldata/learned
    b = ball[:, :2].copy()
    seen = np.isfinite(b[:, 0])
    P = picks(min_conf)
    pf = np.array(sorted(P), int)
    pxy = np.array([P[f][:2] for f in pf], np.float64).reshape(-1, 2)
    filled = np.zeros(len(b), bool)
    for a, z in halves:
        m = ~seen[a:z + 1]
        d = np.diff(np.r_[0, m.astype(int), 0])
        for s, e in zip(np.flatnonzero(d == 1) + a, np.flatnonzero(d == -1) + a):
            if max_gap is not None and e - s > max_gap:
                continue
            k = (pf >= s) & (pf < e)
            if not k.any():
                continue
            fr, xy = pf[k], pxy[k]
            if s - 1 >= a and seen[s - 1]:
                fr, xy = np.r_[s - 1, fr], np.r_[b[s - 1:s], xy]
            if e <= z and seen[e]:
                fr, xy = np.r_[fr, e], np.r_[xy, b[e:e + 1]]
            fr, xy = _drop_jumps(fr, xy)
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
    if smooth:  # the tracker's ball is a 0.7 Hz low-pass (alldata/robust); give the fill the same
        lp = feats.lowpass(b, 5, 0.7, load.FPS)
        b[filled] = lp[filled]
    sys.path.insert(0, os.path.join(ROOT, "external"))
    from build import speed
    out = np.concatenate([b, speed(b, load.FPS)[:, None]], -1).astype(ball.dtype)
    return out, filled
