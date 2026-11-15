"""What the labels and the ball say about stoppages and restarts, as extra TRAINING targets.

Nothing here is ever an input. For every half of an outside match, at the model's 5 fps frames:

    rb     (T, 2)  where the player who takes the next restart stands, on dead frames in the last
                   WINDOW seconds before play resumes (NaN elsewhere). The taker is the tracked
                   player nearest the ball on the first live frame; the net's pointer is taught to
                   find that player while the ball is still dead. Canonical pitch, as data.py.
    rk     (T,)    kind of the next restart on those frames: 0 throw-in, 1 corner, 2 goal kick,
                   3 kick-off, 4 free kick or other; -1 elsewhere or where the ball was not seen.
    rt     (T, 2)  log(1 + seconds) to the next change between live and dead, and since the last
                   one, capped at CAP seconds; NaN where an ungraded frame lies in between.

The restart kind is read off where the ball is when play resumes, the usual rule in the event
detection literature (Vidal-Codina et al. 2022, "Automatic event detection in football using
tracking data"), which then confirms it from the players' layout; here the net has to find that
layout itself, because at prediction it has the players only.
"""

import os
import pickle

import numpy as np

import data as D

WINDOW = 6.0   # seconds before a restart on which the taker and the kind are taught
CAP = 15.0     # seconds; the timing targets saturate here
REACH = 3.0    # metres: the taker must be this near the ball when play resumes
STAY = 6.0     # metres: and is pointed at only while this near the restart spot
KINDS = ("throw-in", "corner", "goal kick", "kick-off", "free kick")


def kind_of(x, y, L, W):
    """x, y in metres from the pitch centre."""
    x, y = abs(x), abs(y)
    if x > L / 2 - 3 and y > W / 2 - 3:
        return 1
    if y > W / 2 - 1.5:
        return 0
    if x > L / 2 - 8 and y < 12:
        return 2
    if x < 2 and y < 2:
        return 3
    return 4


def half_targets(players, ball, team, labels, a, b, pitch, fps=25):
    L, W = pitch
    lab = labels[a:b + 1]
    n = len(lab)
    centre = np.array([L / 2, W / 2], np.float32)
    sign = 1.0 if D.home_defends_left(players, team, a, b) else -1.0
    xy = (players[a:b + 1, :, :2].astype(np.float32) - centre) * sign
    bxy = (ball[a:b + 1, :2].astype(np.float32) - centre) * sign
    dead = lab == "D"
    live = (lab == "H") | (lab == "A")
    rb = np.full((n, 2), np.nan, np.float32)
    rk = np.full(n, -1, np.int64)
    w = int(WINDOW * fps)
    for f in np.flatnonzero(live[1:] & dead[:-1]) + 1:   # first live frame after a stoppage
        seen = np.flatnonzero(np.isfinite(bxy[f:f + fps, 0]))
        if not len(seen):
            continue
        g = f + seen[0]
        spot = bxy[g]
        s = f - 1
        while s > 0 and dead[s - 1] and f - s < w:
            s -= 1
        rk[s:f] = kind_of(spot[0], spot[1], L, W)
        dist = np.linalg.norm(xy[g] - spot, axis=-1)
        if not np.isfinite(dist).any() or np.nanmin(dist) > REACH:
            continue
        k = int(np.nanargmin(dist))
        track = xy[s:f, k]
        near = np.linalg.norm(track - spot, axis=-1) < STAY
        rb[s:f][near] = track[near]
    # seconds to the next and since the last live/dead change
    state = np.where(dead, 1, np.where(live, 0, -1))
    flip = np.flatnonzero((state[1:] != state[:-1]) & (state[1:] >= 0) & (state[:-1] >= 0)) + 1
    bad = np.flatnonzero(state < 0)
    t = np.arange(n)

    def gap(marks, ahead):
        if not len(marks):
            return np.full(n, np.inf)
        i = np.searchsorted(marks, t, side="left" if ahead else "right")
        if ahead:
            return np.where(i < len(marks), marks[np.minimum(i, len(marks) - 1)] - t, np.inf)
        return np.where(i > 0, t - marks[np.maximum(i - 1, 0)], np.inf)

    rt = np.full((n, 2), np.nan, np.float32)
    for j, ahead in enumerate((True, False)):
        to_flip, to_bad = gap(flip, ahead), gap(bad, ahead)
        ok = (state >= 0) & (to_flip < to_bad)   # no ungraded frame before the change
        cap_ok = (state >= 0) & (np.minimum(to_flip, to_bad) >= CAP * fps)   # far from both: capped, still known
        rt[ok | cap_ok, j] = np.log1p(np.minimum(to_flip[ok | cap_ok] / fps, CAP))
    idx = np.arange(0, n, D.STEP)
    return dict(rb=rb[idx] / np.array([52.5, 34.0], np.float32), rk=rk[idx], rt=rt[idx])


def load_all():
    """{match: [targets per half]}, cached."""
    cache = os.path.join(D.ROOT, "data", "cache", "deep_restarts.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    out = {}
    for p in D.external_paths():
        d = np.load(p)
        out[os.path.basename(p)[:-4]] = [
            half_targets(d["players"], d["ball"], d["team"], d["labels"], int(a), int(b), tuple(d["pitch"]), int(d["fps"]))
            for a, b in d["halves"]]
    with open(cache + f".{os.getpid()}", "wb") as f:
        pickle.dump(out, f, protocol=4)
    os.replace(cache + f".{os.getpid()}", cache)
    return out
