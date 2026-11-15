"""Loading for alldata/physics: the outside matches and BAR-ATH in one shape, the ball as the model
is shown it, and the players-only model's probabilities used as a prior where the ball is missing.

The ball the model is shown ("shown") is not always the ball as recorded:

- IDSSE freezes the ball while play is dead (it moves exactly zero), which no automatic tracker
  does. A frozen ball is treated as not observed, on every run.
- mode "barath" shows the resting ball in 40% of IDSSE's stoppages (BAR-ATH's tracker does see a
  dead ball: 15% of its ball-seen frames in the public window are dead), with a little jitter,
  and hides the ball the way BAR-ATH loses it in live play: in long gaps (BAR-ATH:
  99% of its missing frames sit in gaps over 5 s, median 10 s), not in the random 1-2 s dropouts
  `external/score.py --seen` makes. About 12% of live frames are hidden, which is the Metrica
  samples' own rate (9-13%) and what the players-only model implies for BAR-ATH. The Metrica
  samples already lose the ball this way and keep their own gaps.
- mode "barath" then smooths every ball path the way BAR-ATH's is smoothed. BAR-ATH's ball takes
  a second to reach full speed after a kick. Its acceleration around an isolated kick, which
  should be one spike, is a rounded bump 1.5 s wide: zero beyond 20 frames either side, 60% of
  its peak at 10 frames, standard deviation 7.4 frames. That is a parabolic window of half-width
  about 16 frames (0.64 s; 14 to 18 reproduce its acceleration quantiles on IDSSE). Such a window has no usable inverse (its spectrum has zeros from
  1 Hz), so the outside matches are smoothed to match rather than BAR-ATH sharpened. Without
  this a touch detector tuned on the outside matches finds a fifth of the touches on BAR-ATH.
"""

import glob
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation")]

CACHE = os.path.join(ROOT, "data", "cache", "alldata")
PRIOR_DIR = os.path.join(ROOT, "data", "cache", "deep")  # deep/train.py's fold probabilities
STEP = 5  # features and probabilities every 5th frame (5 Hz), as deep/
PROVIDERS = ("idsse", "metrica", "skillcorner")

FROZEN_SECONDS = 0.6      # a ball that has not moved at all for this long is IDSSE's dead-ball freeze
LIVE_GAP_SHARE = 0.12     # share of live frames hidden in mode "barath"
LIVE_GAP_MEDIAN = 7.0     # seconds; lognormal, sigma 0.8
LIVE_GAP_SIGMA = 0.8


def names():
    return [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))]


def provider(name):
    return name.split("_")[0]


def load_match(name):
    z = np.load(os.path.join(ROOT, "data", "external", name + ".npz"))
    d = {k: z[k] for k in z.files}
    d["halves"] = [tuple(int(v) for v in h) for h in d["halves"]]
    d["fps"] = int(d["fps"])
    d["pitch"] = tuple(float(v) for v in d["pitch"])
    d["name"] = name
    return d


def load_bar_ath():
    sys.path.insert(0, os.path.join(ROOT, "possession"))
    import load as L
    players, ball, team, halves = L.load()
    return dict(players=players, ball=ball, team=team, halves=[tuple(int(v) for v in h) for h in halves],
                pitch=tuple(L.PITCH), fps=int(L.FPS), labels=None, name="bar_ath")


def frozen(ball, fps):
    """Frames inside a run where the recorded ball does not move at all for FROZEN_SECONDS or more."""
    same = np.zeros(len(ball), bool)
    same[1:] = (ball[1:, 0] == ball[:-1, 0]) & (ball[1:, 1] == ball[:-1, 1])
    out = np.zeros(len(ball), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, same.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= FROZEN_SECONDS * fps:
            out[max(a - 1, 0):b] = True
    return out


DEAD_SHOWN_SHARE = 0.4    # share of IDSSE's stoppages whose resting ball is shown in mode "barath"
SMOOTH_FRAMES = (14.0, 18.0)  # half-width of the smoothing window in 25 fps frames, drawn per match
REST_JITTER = 0.02        # m, so a shown resting ball is not exactly still


def shown_ball(d, mode="native", seed=0):
    """The ball as the model is shown it: (n, 3) with NaN where hidden. mode: native | barath |
    recorded (see the module docstring)."""
    ball = d["ball"].astype(np.float64).copy()
    prov = provider(d["name"])
    fps = d["fps"]
    n = len(ball)
    rng = np.random.default_rng(zlib.crc32(f"{d['name']} {seed}".encode()))
    hide = np.zeros(n, bool)
    if prov == "idsse" and mode != "recorded":
        fz = frozen(ball, fps)
        hide |= fz
        if mode == "barath":
            # some stoppages keep their resting ball in view: all of it, or the seconds before the restart
            edges = np.flatnonzero(np.diff(np.r_[0, fz.astype(np.int8), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if rng.random() < DEAD_SHOWN_SHARE:
                    s = a if rng.random() < 0.5 else max(a, b - int(fps * rng.uniform(1, 10)))
                    hide[s:b] = False
                    ball[s:b, :2] += REST_JITTER * rng.standard_normal((b - s, 2))
    if mode == "barath" and prov in ("idsse", "skillcorner"):
        mean_len = LIVE_GAP_MEDIAN * np.exp(LIVE_GAP_SIGMA ** 2 / 2) * fps
        rate = LIVE_GAP_SHARE / mean_len  # gap starts per frame
        for s in np.flatnonzero(rng.random(n) < rate):
            ln = int(fps * LIVE_GAP_MEDIAN * np.exp(LIVE_GAP_SIGMA * rng.standard_normal()))
            hide[s:s + ln] = True
    ball[hide] = np.nan
    if mode == "barath":
        ball = smooth(ball, rng.uniform(*SMOOTH_FRAMES) * fps / 25)
    return ball


def smooth(ball, half_width):
    """The shown ball's path smoothed by a parabolic (Epanechnikov) window of this half-width in
    frames, renormalised at the edges of a gap."""
    h = int(np.ceil(half_width))
    t = np.arange(-h, h + 1)
    k = np.clip(1 - (t / half_width) ** 2, 0, None)
    ok = np.isfinite(ball[:, 0])
    w = np.convolve(ok.astype(float), k, "same")
    out = ball.copy()
    for c in (0, 1):
        out[:, c] = np.convolve(np.where(ok, ball[:, c], 0.0), k, "same") / np.maximum(w, 1e-9)
    out[~ok] = np.nan
    return out


def shown_players(d, seed=0):
    """Players lost the way BAR-ATH's tracker loses them (external/score.py drop_players: about 7%
    of player-frames, in runs of seconds and in bursts), under this folder's own seed."""
    sys.path.insert(0, os.path.join(ROOT, "external"))
    from score import drop_players
    return drop_players(d["players"], d["halves"], d["fps"], f"alldata {d['name']} {seed}")


def prior_probs(name, tags=("bt",)):
    """The players-only model's held-out probabilities for one outside match (its provider's fold,
    so the network never trained on this provider), per half at 5 Hz; several tags are averaged."""
    out = None
    for t in tags:
        path = os.path.join(PRIOR_DIR, t, "bar_ath_probs.npz") if name == "bar_ath" else \
            os.path.join(PRIOR_DIR, t, provider(name), name + ".npz")
        z = np.load(path)
        ps = [z[k].astype(np.float64) for k in sorted(k for k in z.files if k.startswith("p"))]
        out = ps if out is None else [a + b for a, b in zip(out, ps)]
    return [p / len(tags) for p in out]


def swap_teams(d, prior):
    """The same match with home and away exchanged (labels, team ids, prior columns)."""
    e = dict(d)
    e["team"] = 1 - d["team"]
    if d.get("labels") is not None:
        lab = d["labels"]
        e["labels"] = np.where(lab == "H", "A", np.where(lab == "A", "H", lab))
    return e, [p[:, [1, 0, 2]] for p in prior]
