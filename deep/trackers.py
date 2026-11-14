"""Make one tracking system's output look like another's. Players only: nothing here reads the ball.

Each function takes positions xy (n, K, 2) in metres at `fps`, NaN where a player is not tracked,
and returns a changed copy. They stand for the ways trackers differ that a model could latch on
to: measurement noise, how hard the provider smooths, the native frame rate behind a 25 fps
export, positions snapped to a grid, a calibration that is slightly off, and players the camera
or the tracker does not see.

`style(rng)` draws a random mix of them for training (deep/train.py --style); `shift.py` applies
them one at a time to held-out matches to see which of them an existing model is sensitive to.
"""

import numpy as np


def white(xy, rng, sigma):
    """Independent noise per frame, as a detector's box jitter."""
    return xy + rng.normal(0, sigma, xy.shape).astype(np.float32)


def drift(xy, rng, sigma, seconds, fps=25):
    """Slow per-player error (a track that sits off the player for a while): noise with a
    correlation time of `seconds`, built at a coarse step and interpolated."""
    n, K, _ = xy.shape
    step = max(1, int(seconds * fps))
    knots = rng.normal(0, sigma, (n // step + 2, K, 2)).astype(np.float32)
    t = np.arange(n) / step
    i = t.astype(int)
    w = (t - i)[:, None, None].astype(np.float32)
    return xy + knots[i] * (1 - w) + knots[i + 1] * w


def smooth(xy, seconds, fps=25):
    """Centred moving average over `seconds`, skipping missing frames."""
    k = max(1, int(round(seconds * fps)) | 1)
    if k == 1:
        return xy.copy()
    ok = np.isfinite(xy[..., :1])
    c = np.cumsum(np.concatenate([np.zeros_like(xy[:1]), np.where(ok, xy, 0)]), 0)
    m = np.cumsum(np.concatenate([np.zeros((1,) + ok.shape[1:], np.float32), ok.astype(np.float32)]), 0)
    n, h = len(xy), k // 2
    lo, hi = np.clip(np.arange(n) - h, 0, n), np.clip(np.arange(n) + h + 1, 0, n)
    out = (c[hi] - c[lo]) / np.maximum(m[hi] - m[lo], 1)
    return np.where(ok, out, np.nan).astype(np.float32)


def resample(xy, rate, fps=25, phase=0.0):
    """As if tracked at `rate` frames a second and exported at `fps` by linear interpolation
    (SkillCorner is 10 fps underneath). Missing at either neighbour stays missing."""
    n = len(xy)
    step = fps / rate
    k = (np.arange(n) - phase) / step
    i0 = np.clip(np.floor(k), 0, None)
    a = np.clip(np.round(i0 * step + phase).astype(int), 0, n - 1)
    b = np.clip(np.round((i0 + 1) * step + phase).astype(int), 0, n - 1)
    w = np.clip((np.arange(n) - a) / np.maximum(b - a, 1), 0, 1)[:, None, None].astype(np.float32)
    return xy[a] * (1 - w) + xy[b] * w


def grid(xy, gx, gy):
    """Positions snapped to a grid (SoccerTrack v2's are on 1.05 m x 0.68 m)."""
    g = np.array([gx, gy], np.float32)
    return np.round(xy / g) * g


def affine(xy, centre, sx=1.0, sy=1.0, dx=0.0, dy=0.0, turn=0.0):
    """A calibration that is a little off: scale, shift and turn about the centre spot."""
    c = np.asarray(centre, np.float32)
    p = (xy - c) * np.array([sx, sy], np.float32)
    co, si = np.cos(turn), np.sin(turn)
    q = np.stack([p[..., 0] * co - p[..., 1] * si, p[..., 0] * si + p[..., 1] * co], -1)
    return (q + c + np.array([dx, dy], np.float32)).astype(np.float32)


def lose(xy, rng, share, seconds, fps=25):
    """Each player lost in runs of mean `seconds`, `share` of the time."""
    n, K, _ = xy.shape
    out = xy.copy()
    if share <= 0:
        return out
    off = seconds * fps
    for k in range(K):
        i = -int(rng.exponential(off / share))  # start somewhere inside a cycle
        on = True
        while i < n:
            run = 1 + int(rng.exponential(off * (1 - share) / share if on else off))
            if not on and i + run > 0:
                out[max(i, 0):i + run, k] = np.nan
            i, on = i + run, not on
    return out


def view(xy, width, fps=25, lag_seconds=2.0):
    """A broadcast camera: only players within `width` / 2 metres along the pitch of where the
    play is are tracked. Where the play is comes from the players themselves (their mean x,
    smoothed), never from the ball."""
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx = np.nanmean(xy[..., 0], 1)
    cx = np.where(np.isfinite(cx), cx, np.nanmean(cx) if np.isfinite(cx).any() else 0.0)
    cx = smooth(cx[:, None, None].repeat(2, 2).astype(np.float32), lag_seconds, fps)[:, 0, 0]
    out = xy.copy()
    out[np.abs(xy[..., 0] - cx[:, None]) > width / 2] = np.nan
    return out


def style(rng, fps=25):
    """A random tracking system: a list of (name, kwargs) to hand to `apply`. About a quarter of
    draws change nothing, so the model still sees each provider as it is."""
    ops = []
    if rng.random() < 0.25:
        return ops
    if rng.random() < 0.3:
        ops.append(("affine", dict(sx=rng.uniform(0.96, 1.04), sy=rng.uniform(0.96, 1.04),
                                   dx=rng.normal(0, 0.7), dy=rng.normal(0, 0.7), turn=rng.normal(0, 0.01))))
    if rng.random() < 0.3:
        ops.append(("drift", dict(sigma=rng.uniform(0.1, 0.6), seconds=rng.uniform(1, 5))))
    if rng.random() < 0.4:
        ops.append(("white", dict(sigma=rng.uniform(0.02, 0.3))))
    if rng.random() < 0.3:
        ops.append(("resample", dict(rate=rng.choice([5, 6.25, 10, 12.5]), phase=float(rng.integers(5)))))
    if rng.random() < 0.15:
        ops.append(("grid", dict(gx=rng.uniform(0.1, 1.1), gy=rng.uniform(0.1, 0.8))))
    if rng.random() < 0.5:
        ops.append(("smooth", dict(seconds=rng.uniform(0.1, 1.2))))
    if rng.random() < 0.15:
        ops.append(("view", dict(width=rng.uniform(45, 80))))
    if rng.random() < 0.3:
        ops.append(("lose", dict(share=rng.uniform(0.01, 0.12), seconds=rng.uniform(1, 8))))
    return ops


def apply(xy, ops, rng, centre, fps=25):
    for name, kw in ops:
        if name == "white":
            xy = white(xy, rng, **kw)
        elif name == "drift":
            xy = drift(xy, rng, fps=fps, **kw)
        elif name == "smooth":
            xy = smooth(xy, fps=fps, **kw)
        elif name == "resample":
            xy = resample(xy, fps=fps, **kw)
        elif name == "grid":
            xy = grid(xy, **kw)
        elif name == "affine":
            xy = affine(xy, centre, **kw)
        elif name == "lose":
            xy = lose(xy, rng, fps=fps, **kw)
        elif name == "view":
            xy = view(xy, fps=fps, **kw)
        else:
            raise ValueError(name)
    return xy
