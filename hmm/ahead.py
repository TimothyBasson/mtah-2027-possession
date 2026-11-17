"""Look-ahead ball clue: where a missing ball turns up again.

The place clue (hmm/clues.py ball_place_category) reads a missing ball by where it was last seen.
On BAR-ATH most false stoppages are the other half of that story: the ball vanishes in play near a
goal (a save, a cross gathered) and is not tracked while the keeper holds it, the players slow
down, and the decoder calls it dead until the keeper releases it. Where the ball comes back says
what happened: in the hands of the keeper, inside his own box, means he had it all along; on or
past a line means it went out. The decoding is offline, so the next sighting is fair to read.

ahead_category gives every missing frame a category from (where the ball was last seen, where it
is next seen); a seen frame gets none (-1). The table is p(next | last, state), so it adds only
what the next sighting says beyond the place clue that already reads the last one. Keepers come
from the slots (BAR-ATH metadata) or, on the external matches, from where the players stand.
"""

import warnings

import numpy as np

# where the ball was last seen before the gap
BACK_LINE, BACK_PLAY = 0, 1          # within LINE_MARGIN of a line or past it / anywhere else or never
# where it turns up next
NEXT_LINE, NEXT_KEEPER_HOME, NEXT_KEEPER_AWAY, NEXT_PLAY, NEXT_NONE = range(5)
N_NEXT = 5
N_AHEAD = 2 * N_NEXT

LINE_MARGIN = 2.0      # metres, as hmm/clues.py
KEEPER_REACH = 2.0     # metres: the ball this close to the keeper is in his hands or at his feet
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16   # the penalty area, metres
AHEAD_SECONDS = 30.0   # a sighting further ahead than this says nothing about the gap


def _outside(bx, by, pitch):
    length, width = pitch
    return np.maximum.reduce([-bx, bx - length, -by, by - width])


def keepers_from_positions(px, py, team, halves, length):
    """(frames,) slot of the home and the away keeper per frame: per half, the player of each team
    standing deepest on average towards that team's own goal (the end its players stand nearer)."""
    n = len(px)
    out = np.full((n, 2), -1, int)
    for lo, hi in halves:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx = np.nanmean(px[lo:hi], axis=0)
            tracked = np.mean(np.isfinite(px[lo:hi]), axis=0)
        for t in (0, 1):
            mine = (team == t) & (tracked > 0.3)
            if not mine.any():
                continue
            low_end = np.nanmean(mx[mine]) < length / 2
            cand = np.where(mine)[0]
            k = cand[np.argmin(mx[cand])] if low_end else cand[np.argmax(mx[cand])]
            out[lo:hi, t] = k
    return out


def ahead_category(bx, by, usable, px, py, keeper_slot, pitch, fps, halves):
    """Category per frame (BACK_* * N_NEXT + NEXT_*), -1 where the ball is seen or outside the halves.

    keeper_slot is (frames, 2): the home and away keeper's slot on each frame, -1 if unknown."""
    length, width = pitch
    n = len(bx)
    cats = np.full(n, -1, np.int8)
    out = _outside(bx, by, pitch)
    horizon = int(AHEAD_SECONDS * fps)
    frames = np.arange(n)
    for lo, hi in halves:
        u = usable[lo:hi]
        f = frames[lo:hi]
        last = np.maximum.accumulate(np.where(u, f, -1))
        nxt = np.minimum.accumulate(np.where(u, f, hi + horizon + 1)[::-1])[::-1]
        miss = ~u
        back = np.full(hi - lo, BACK_PLAY, np.int8)
        has_last = miss & (last >= 0)
        back[has_last] = np.where(out[last[has_last]] >= -LINE_MARGIN, BACK_LINE, BACK_PLAY)
        nx = np.full(hi - lo, NEXT_NONE, np.int8)
        ok = miss & (nxt < hi) & (nxt - f <= horizon)
        g = nxt[ok]
        kind = np.full(len(g), NEXT_PLAY, np.int8)
        for side, code in ((0, NEXT_KEEPER_HOME), (1, NEXT_KEEPER_AWAY)):
            k = keeper_slot[g, side]
            valid = k >= 0
            kx = np.where(valid, px[g, np.maximum(k, 0)], np.nan)
            ky = np.where(valid, py[g, np.maximum(k, 0)], np.nan)
            with np.errstate(invalid="ignore"):
                near = np.hypot(kx - bx[g], ky - by[g]) <= KEEPER_REACH
                in_box = (np.minimum(kx, length - kx) <= BOX_DEPTH) & (np.abs(ky - width / 2)
                                                                        <= BOX_HALF_WIDTH)
            kind[near & in_box & (kind == NEXT_PLAY)] = code
        kind[out[g] >= -LINE_MARGIN] = NEXT_LINE
        nx[ok] = kind
        c = back * N_NEXT + nx
        cats[lo:hi][miss] = c[miss]
    return cats


def mirror(table):
    """Home and away keeper columns swapped, for the last axis."""
    t = np.array(table, float)
    idx = np.arange(N_AHEAD).reshape(2, N_NEXT)
    idx[:, [NEXT_KEEPER_HOME, NEXT_KEEPER_AWAY]] = idx[:, [NEXT_KEEPER_AWAY, NEXT_KEEPER_HOME]]
    return t[..., idx.ravel()]


def counts(cats, labels):
    """(home, dead) counts of each category: away frames counted mirrored into home, dead frames
    both ways round (so neither row favours a team)."""
    c = lambda sel: np.bincount(cats[sel & (cats >= 0)], minlength=N_AHEAD).astype(float)  # noqa
    home = c(labels == "H") + mirror(c(labels == "A"))
    dead = c(labels == "D")
    dead = (dead + mirror(dead)) / 2
    return home, dead


def table(home, dead, prior=1.0):
    """log p(next | last, state) as a (3, N_AHEAD) table from summed counts, normalised within each
    last-seen group, with `prior` pseudo-counts per cell."""
    rows = []
    for r in (home, mirror(home), dead):
        r = (np.asarray(r, float) + prior).reshape(2, N_NEXT)
        rows.append((r / r.sum(axis=1, keepdims=True)).ravel())
    return np.log(np.array(rows))
