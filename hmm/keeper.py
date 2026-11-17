"""Keeper-area ball clue: the ball missing or still in a keeper's hands, inside his own box.

hmm/ahead.py reads every missing-ball gap by where the ball turns up next. At four times its
weight it removes most of the BAR-ATH false stoppages where the keeper holds the ball, but its
line and open-play categories also move stoppages elsewhere, and the held-out mean drops. This
clue keeps only the keeper part and says nothing anywhere else:

  KEEP_MISS_HOME / _AWAY   the ball is missing, and the gap starts or ends with the ball within
                           KEEPER_REACH of that team's keeper, inside his own box (he caught it,
                           or he releases it), and the gap is not ended by the ball on a line;
  KEEP_STILL_HOME / _AWAY  the ball is seen, slower than STILL_SPEED, within KEEPER_REACH of that
                           keeper inside his own box (he holds it, or it lies at his feet).

Every other frame gets -1 and no term. The table is p(category | state) over all frames, counted
on labelled matches with the home/away mirror (see `counts`), so it is evidence for the keeper's
team against dead ball, not a hand-set rule.
"""

import numpy as np

from hmm.ahead import AHEAD_SECONDS, BOX_DEPTH, BOX_HALF_WIDTH, LINE_MARGIN, _outside

KEEP_MISS_HOME, KEEP_MISS_AWAY, KEEP_STILL_HOME, KEEP_STILL_AWAY = range(4)
N_KEEP = 4
NONE = N_KEEP          # the column every other frame is counted in

KEEPER_REACH = 2.0     # metres
STILL_SPEED = 2.0      # m/s: a held or rolled ball, not a pass or a shot


def _speed(bx, by, fps, smooth=0.4):
    v = np.hypot(np.gradient(bx), np.gradient(by)) * fps
    w = max(1, int(round(smooth * fps)))
    ok = np.isfinite(v)
    num = np.convolve(np.where(ok, v, 0), np.ones(w), "same")
    den = np.convolve(ok.astype(float), np.ones(w), "same")
    return np.where(den > w / 2, num / np.maximum(den, 1), np.nan)


def at_keeper(bx, by, px, py, keeper_slot, pitch, frames):
    """(len(frames), 2) bool: the ball within KEEPER_REACH of the home / away keeper, the keeper
    inside a penalty area."""
    length, width = pitch
    out = np.zeros((len(frames), 2), bool)
    for side in (0, 1):
        k = keeper_slot[frames, side]
        valid = k >= 0
        kx = np.where(valid, px[frames, np.maximum(k, 0)], np.nan)
        ky = np.where(valid, py[frames, np.maximum(k, 0)], np.nan)
        with np.errstate(invalid="ignore"):
            near = np.hypot(kx - bx[frames], ky - by[frames]) <= KEEPER_REACH
            in_box = ((np.minimum(kx, length - kx) <= BOX_DEPTH)
                      & (np.abs(ky - width / 2) <= BOX_HALF_WIDTH))
        out[:, side] = near & in_box
    return out


def in_keeper_box(bx, by, px, py, keeper_slot, pitch, frames):
    """(len(frames), 2) bool: the ball inside the penalty area at the end where the home / away
    keeper stands."""
    length, width = pitch
    out = np.zeros((len(frames), 2), bool)
    for side in (0, 1):
        k = keeper_slot[frames, side]
        valid = k >= 0
        kx = np.where(valid, px[frames, np.maximum(k, 0)], np.nan)
        b = bx[frames]
        with np.errstate(invalid="ignore"):
            same_end = (kx < length / 2) == (b < length / 2)
            out[:, side] = (valid & same_end & (np.minimum(b, length - b) <= BOX_DEPTH)
                            & (np.minimum(b, length - b) >= 0)
                            & (np.abs(by[frames] - width / 2) <= BOX_HALF_WIDTH))
    return out


def keeper_category(bx, by, usable, px, py, keeper_slot, pitch, fps, halves, use=(0, 1, 2, 3),
                    ends="both"):
    """Category per frame (KEEP_*), -1 elsewhere. `use` keeps a subset of the categories.

    `ends` says which missing-ball gaps count for a keeper (the last sighting always in play, not
    on or past a line, so a ball out for a goal kick never counts):
      "next"  the ball turns up at him (in his hands, inside his box), not on a line;
      "last"  it was lost at him;
      "both"  either;
      "box"   lost in play inside his box, and it turns up at him: a save or a gathered cross
              that he holds while the tracker cannot see it."""
    n = len(bx)
    cats = np.full(n, -1, np.int8)
    out = _outside(bx, by, pitch)
    speed = _speed(np.where(usable, bx, np.nan), np.where(usable, by, np.nan), fps)
    horizon = int(AHEAD_SECONDS * fps)
    frames = np.arange(n)
    for lo, hi in halves:
        u = usable[lo:hi]
        f = frames[lo:hi]
        last = np.maximum.accumulate(np.where(u, f, -1))
        nxt = np.minimum.accumulate(np.where(u, f, hi + horizon + 1)[::-1])[::-1]
        miss = ~u
        gap = miss & (last >= 0) & (f - last <= horizon) & (nxt < hi) & (nxt - f <= horizon)
        gap[gap] &= (out[last[gap]] < -LINE_MARGIN) & (out[nxt[gap]] < -LINE_MARGIN)
        c = np.full(hi - lo, -1, np.int8)
        lk = np.zeros((hi - lo, 2), bool)
        nk = np.zeros((hi - lo, 2), bool)
        lb = np.zeros((hi - lo, 2), bool)
        lk[gap] = at_keeper(bx, by, px, py, keeper_slot, pitch, last[gap])
        nk[gap] = at_keeper(bx, by, px, py, keeper_slot, pitch, nxt[gap])
        lb[gap] = in_keeper_box(bx, by, px, py, keeper_slot, pitch, last[gap])
        hit = {"next": nk, "last": lk, "both": nk | lk, "box": lb & nk}[ends]
        seen = u & (speed[lo:hi] < STILL_SPEED)
        st = np.zeros((hi - lo, 2), bool)
        st[seen] = at_keeper(bx, by, px, py, keeper_slot, pitch, f[seen])
        for side in (0, 1):
            c[hit[:, side] & (c < 0)] = KEEP_MISS_HOME + side
            c[st[:, side] & (c < 0)] = KEEP_STILL_HOME + side
        c[~np.isin(c, use)] = -1
        cats[lo:hi] = c
    return cats


def mirror(table):
    """Home and away columns swapped on the last axis."""
    t = np.array(table, float)
    return t[..., [KEEP_MISS_AWAY, KEEP_MISS_HOME, KEEP_STILL_AWAY, KEEP_STILL_HOME, NONE]]


def counts(cats, labels):
    """(home, dead) counts of each category plus NONE, over the H/A/D frames: away frames mirrored
    into home, dead frames counted both ways round."""
    def c(sel):
        x = np.where(cats >= 0, cats, NONE)[sel]
        return np.bincount(x, minlength=N_KEEP + 1).astype(float)
    home = c(labels == "H") + mirror(c(labels == "A"))
    dead = c(labels == "D")
    return home, (dead + mirror(dead)) / 2


def table(home, dead, prior=1.0):
    """log p(category | state), states (H, A, D) x N_KEEP categories, from (home, dead) counts with
    `prior` pseudo-counts per cell. The NONE column only normalises."""
    rows = []
    for r in (home, mirror(home), dead):
        r = np.asarray(r, float) + prior
        rows.append(r / r.sum())
    return np.log(np.array(rows))[:, :N_KEEP]
