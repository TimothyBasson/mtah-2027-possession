"""Make an IDSSE or Metrica match look like SkillCorner's broadcast tracking, for training only.

    python external/broadcast.py [--tag bc] [ball_sigma=0.08 ...]   # -> data/cache/control_<tag>+<match>.npz
    python external/train_control.py --loso --augment bc [--augment-mode add] --out control_loso_bc.npz
    python external/score_control.py --loso control_loso_bc.npz

The control classifier (hmm/control.py) is trained on optical 25 fps tracking (IDSSE, Metrica) and
is weakest on held-out SkillCorner, which is broadcast tracking at 10 fps, linearly interpolated to
25 (external/build.py). Its touch features read the ball's velocity change, which that
interpolation distorts. After PathCRF (arXiv 2602.12080), this module degrades the optical matches
the same way, so the classifier can be trained on rows that look like the held-out provider:

1. 10 fps: positions are sampled every 2.5 frames and put back at 25 fps with the same linear
   interpolation build.py uses for SkillCorner (NaN between two samples when either is NaN).
2. Jitter: Gaussian noise on the 10 fps samples, before the interpolation, so it is correlated
   across 25 fps frames the way SkillCorner's is. Ball and players have their own sigma.
3. Ball dropout in bursts: two-state on/off runs (as external/score.drop_ball) down to
   SkillCorner's own share seen, before control_features.match_ball tops every set up to 0.59.
4. Players far from the ball dropped in bursts: each player beyond `far_metres` of the ball is
   lost with a hazard that grows with the distance, for runs of mean `far_seconds`.

The labels are not touched: they are folded with the ball as recorded (external/control.py).
The sizes were set from SkillCorner's tracking alone (no labels): see the README.
"""

import glob
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
from control_features import CACHE, SEEN  # noqa: E402
from hmm.control import features  # noqa: E402
from score import drop_ball  # noqa: E402

SOURCES = ("idsse", "metrica")
DEFAULTS = dict(
    fps10=1,            # 1: resample through 10 fps; 0: leave at 25 fps
    ball_sigma=0.08,    # metres, per axis, on the 10 fps ball samples
    player_sigma=0.0,   # metres, per axis, on the 10 fps player samples
    ball_seen=0.81,     # share of frames the ball is seen after the bursts (SkillCorner: 0.81)
    ball_gap=3.3,       # mean burst length, seconds (SkillCorner: 3.3)
    far_metres=0.0,     # players beyond this distance of the ball can be lost; 0: nobody is
    far_share=0.3,      # share of the time a player 20 m beyond far_metres is lost
    far_seconds=3.0,    # mean length of a lost run, seconds
)
SC_FPS = 10


def to_10fps(x, fps):
    """Samples of x (frames, ...) every fps / 10 frames, linearly interpolated; NaN when either
    neighbour is NaN, but a sample that lands exactly on a frame is kept."""
    n = int((len(x) - 1) * SC_FPS // fps) + 1
    t = np.arange(n) * fps / SC_FPS
    i = np.minimum(t.astype(int), len(x) - 2)
    w = (t - i).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return np.where(w == 0, x[i], x[i] * (1 - w) + x[i + 1] * w)


def to_25fps(x, n, fps):
    """external/build.resample: 10 fps samples back to n frames at fps."""
    t = np.arange(n) * SC_FPS / fps
    i = np.minimum(t.astype(int), len(x) - 2)
    w = (t - i).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return np.where(w == 0, x[i], x[i] * (1 - w) + x[i + 1] * w)


def bursts(n, on_len, off_len, rng):
    """(n,) bool, True while on: alternating exponential on / off runs, in frames."""
    keep = np.ones(n, bool)
    i, on = 0, True
    while i < n:
        run = 1 + int(rng.exponential(on_len if on else off_len))
        keep[i:i + run] = on
        i, on = i + run, not on
    return keep


def drop_far_players(xy, ball, fps, rng, far_metres, far_share, far_seconds):
    """xy (frames, K, 2) with players far from the ball lost in runs. A player is at risk while he
    is beyond far_metres of the ball (of its last seen place, while it is unseen); the share of
    time lost rises linearly to far_share at far_metres + 20 m and stays there."""
    out = xy.copy()
    seen = np.isfinite(ball[:, 0])
    last = np.maximum.accumulate(np.where(seen, np.arange(len(ball)), 0))
    b = ball[last]
    dist = np.hypot(xy[..., 0] - b[:, None, 0], xy[..., 1] - b[:, None, 1])
    share = np.clip((dist - far_metres) / 20.0, 0, 1) * far_share
    share = np.where(np.isfinite(share), share, 0)
    off = far_seconds * fps
    for k in range(xy.shape[1]):
        # a run starts with the hazard that gives `share` lost in the long run
        hazard = share[:, k] / np.maximum(1 - share[:, k], 1e-6) / off
        starts = np.flatnonzero(rng.random(len(xy)) < hazard)
        for s in starts:
            out[s:s + 1 + int(rng.exponential(off)), k] = np.nan
    return out


def broadcast_like(d, name, **options):
    """(players_xy, ball_xy) for one external match, degraded as above. Seeded by match name."""
    o = dict(DEFAULTS, **options)
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown options: {}".format(sorted(unknown)))
    rng = np.random.default_rng(zlib.crc32(("broadcast " + name).encode()))
    fps = int(d["fps"])
    xy = d["players"][..., :2].astype(np.float32)
    ball = d["ball"][:, :2].astype(np.float32)
    pxy, bxy = np.empty_like(xy), np.empty_like(ball)
    for a, b in ((int(a), int(b) + 1) for a, b in d["halves"]):
        p, q = xy[a:b], ball[a:b]
        if o["fps10"]:
            p, q = to_10fps(p, fps), to_10fps(q, fps)
        if o["ball_sigma"]:
            q = q + rng.normal(0, o["ball_sigma"], q.shape).astype(np.float32)
        if o["player_sigma"]:
            p = p + rng.normal(0, o["player_sigma"], p.shape).astype(np.float32)
        if o["fps10"]:
            p, q = to_25fps(p, b - a, fps), to_25fps(q, b - a, fps)
        pxy[a:b], bxy[a:b] = p, q
    outside = np.ones(len(xy), bool)
    for a, b in d["halves"]:
        outside[int(a):int(b) + 1] = False
    pxy[outside], bxy[outside] = xy[outside], ball[outside]
    seen = np.isfinite(bxy[:, 0]).mean()
    if o["ball_seen"] < seen:
        off = o["ball_gap"] * fps
        share = o["ball_seen"] / seen
        bxy[~bursts(len(bxy), off * share / (1 - share), off, rng)] = np.nan
    if o["far_metres"]:
        pxy = drop_far_players(pxy, bxy, fps, rng, o["far_metres"], o["far_share"],
                               o["far_seconds"])
    return pxy, bxy


def cache_path(tag, name):
    return os.path.join(CACHE, "control_{}+{}.npz".format(tag, name))


def build(path, tag, options):
    name = os.path.basename(path)[:-4]
    out = cache_path(tag, name)
    if os.path.exists(out):
        return name
    d = np.load(path)
    pxy, bxy = broadcast_like(d, name, **options)
    # as control_features.match_ball: the same ~2 s dropouts down to 0.59 that the decoder is scored on
    bxy = drop_ball(bxy, SEEN, int(d["fps"]))
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    args = (pxy[:, :, 0], pxy[:, :, 1], d["team"], bxy[:, 0].astype(float),
            bxy[:, 1].astype(float), tuple(float(v) for v in d["pitch"]), int(d["fps"]), halves)
    X, names = features(*args)
    Xs, _ = features(*args, swap=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, X=X, Xs=Xs, names=np.array(names), options=repr(options))
    return name


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = "bc"
    if "--tag" in args:
        i = args.index("--tag")
        tag = args[i + 1]
        del args[i:i + 2]
    options = {k: float(v) for k, v in (a.split("=") for a in args)}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz"))):
        if os.path.basename(p).startswith(SOURCES):
            t = time.time()
            print(tag, build(p, tag, options), "%.0f s" % (time.time() - t), flush=True)
