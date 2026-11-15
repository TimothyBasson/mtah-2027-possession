"""Give back the players a tracker lost for a short while, before any model reads the tracking.

    players2, team2 = impute.fill(players, team, halves, fps, max_gap=15)

BAR-ATH's tracker loses a player for seconds at a time, mostly while he stands next to another
(adapt/dropout.py), and brings him back in whichever of its 46 slots is free. The outside matches
the models learn from mostly have all 22 throughout. fill() joins the end of a track to the start
of the next track of the same team that the player could have reached (within 2 m plus 6 m/s of
the gap, the closest first), and puts him on the straight line between the two for the gap. The
task is offline, so looking ahead to where he reappears is allowed. Players only: no ball.

Filled players go in extra slots appended after the tracker's own, so nothing recorded is
overwritten. Gaps longer than max_gap seconds are left empty.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def links(players, team, halves, fps, max_gap, reach=(2.0, 6.0)):
    """[(last tracked frame, slot, first tracked frame again, slot)] for each joined gap."""
    ok = np.isfinite(players[..., 0])
    out = []
    horizon = int(max_gap * fps)
    for a, b in halves:
        for tm in (0, 1):
            ks = np.flatnonzero(team == tm)
            d = np.diff(ok[a:b + 1][:, ks].astype(np.int8), axis=0)
            ev = [(a + f + 1, 1, ks[k]) for f, k in zip(*np.nonzero(d == 1))]
            ev += [(a + f + 1, 0, ks[k]) for f, k in zip(*np.nonzero(d == -1))]
            ev.sort()  # at the same frame a track ending comes before one starting
            lost = []  # (first missing frame, slot, last position)
            for f, born, k in ev:
                lost = [l for l in lost if f - l[0] <= horizon]
                if not born:
                    lost.append((f, k, players[f - 1, k, :2].astype(float)))
                    continue
                p = players[f, k, :2].astype(float)
                best, score = None, 1.0
                for i, (f0, _, p0) in enumerate(lost):
                    s = np.hypot(*(p - p0)) / (reach[0] + reach[1] * (f - f0) / fps)
                    if s <= score:
                        best, score = i, s
                if best is not None:
                    f0, k0, _ = lost.pop(best)
                    out.append((f0 - 1, k0, f, k))
    return out


def fill(players, team, halves, fps, max_gap=15.0):
    """(players with filled gaps in appended slots, team for those slots). Speeds recomputed for
    the appended slots only."""
    sys.path.insert(0, os.path.join(ROOT, "external"))
    from build import speed
    ls = sorted(links(players, team, halves, fps, max_gap))
    if not ls:
        return players, team
    free, teams, cols, keep = [], [], [], []  # per appended slot
    for f0, k0, f1, k1 in ls:
        j = next((j for j, t in enumerate(free) if t <= f0 and teams[j] == team[k0]), None)
        if j is None:
            j = len(free)
            free.append(0)
            teams.append(team[k0])
            cols.append(np.full((len(players), 3), np.nan, np.float32))
            keep.append(np.zeros(len(players), bool))
        w = (np.arange(f0, f1 + 1) - f0)[:, None] / (f1 - f0)
        # written from the last tracked frame to the first again, so his speed is defined through
        # the gap; only the gap itself is kept, since the tracker has him at both ends
        cols[j][f0:f1 + 1, :2] = players[f0, k0, :2] * (1 - w) + players[f1, k1, :2] * w
        keep[j][f0 + 1:f1] = True
        free[j] = f1 + 1
    extra = np.stack(cols, 1)
    extra[..., 2] = speed(extra[..., :2], fps)
    extra[~np.stack(keep, 1)] = np.nan
    return np.concatenate([players, extra], 1), np.concatenate([team, np.array(teams, team.dtype)])
