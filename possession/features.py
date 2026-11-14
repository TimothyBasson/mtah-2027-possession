"""Per-frame measurements from the tracking arrays (possession/load.py, external/build.py)."""

import warnings

import numpy as np

from load import FPS, PITCH  # noqa: F401  (external/score.py repoints PITCH per match)


def nearest_by_team(players, ball, team):
    """Distance in metres from the ball to the nearest home and nearest away player, NaN where
    the ball or that whole side is missing."""
    d = np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        return np.nanmin(d[:, team == 0], 1), np.nanmin(d[:, team == 1], 1)


def own_goal(players, team, halves):
    """Per team, per half: +1 if its own goal is at x=0, -1 if at x=PITCH[0]. The team standing
    nearer x=0 on average defends that end. The two teams are compared with each other, not with
    halfway: a side that dominates a half averages beyond halfway and would be read as defending
    the wrong goal (it happened in a half of 9 of the 30 outside matches)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home = [1 if np.nanmean(players[a:b + 1, team == 0, 0]) < np.nanmean(players[a:b + 1, team == 1, 0]) else -1
                for a, b in halves]
    return [home, [-s for s in home]]


def spread(players, team, halves, keepers=False):
    """Per frame, each team's mean distance from its own centre, in metres: (home, away).

    Without `keepers`, each team's deepest player toward its own goal is left out as the keeper:
    he stands far from everyone, by an amount that depends on where play is, not on who has the
    ball. Which goal is a team's own is read per half from which side its players stand on."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with a side untracked
        for t in (0, 1):
            xy = players[:, team == t, :2].copy()
            if not keepers:
                own = np.zeros(len(xy))
                for (a, b), sign in zip(halves, own_goal(players, team, halves)[t]):
                    own[a:b + 1] = sign
                depth = np.where(np.isfinite(xy[:, :, 0]), xy[:, :, 0] * own[:, None], np.inf)
                rows = np.flatnonzero(np.isfinite(depth).any(1))
                xy[rows, depth[rows].argmin(1)] = np.nan
            out.append(np.nanmean(np.linalg.norm(xy - np.nanmean(xy, 1, keepdims=True), axis=-1), 1))
    return out


def team_signals(players, team, halves, lag=5, max_speed=12.0):
    """(frames, 4) who-has-the-ball signals from the outfield players, each positive when it
    points at home:

    - width:  home's spread across the pitch (sd of y) minus away's. The side with the ball
              stretches the pitch; the side without it narrows to protect the middle.
    - length: the same along the pitch (sd of x).
    - push:   both teams' mean velocity toward the goal home attacks, m/s. Everyone moves the way
              the ball is going: up with the attack, back with the defence.
    - speed:  home's mean speed minus away's. The side chasing the ball works harder.

    Keepers (each team's deepest player) are left out, as in spread()."""
    n = len(players)
    attack = np.zeros(n)  # +1 where home attacks toward x=PITCH[0]
    for (a, b), sign in zip(halves, own_goal(players, team, halves)[0]):
        attack[a:b + 1] = sign
    wid, length, push, pace = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with a side untracked
        for t in (0, 1):
            xy = players[:, team == t, :2].astype(float)
            sp = players[:, team == t, 2].astype(float)
            depth = np.where(np.isfinite(xy[:, :, 0]), xy[:, :, 0] * attack[:, None] * (1 - 2 * t), np.inf)
            rows = np.flatnonzero(np.isfinite(depth).any(1))
            keeper = depth[rows].argmin(1)
            xy[rows, keeper] = sp[rows, keeper] = np.nan
            sp[sp > max_speed] = np.nan
            vx = np.full(xy.shape[:2], np.nan)  # along the pitch, from positions `lag` frames either side
            for a, b in halves:
                vx[a + lag:b + 1 - lag] = (xy[a + 2 * lag:b + 1, :, 0] - xy[a:b + 1 - 2 * lag, :, 0]) * FPS / (2 * lag)
            vx[np.abs(vx) > max_speed] = np.nan
            wid.append(np.nanstd(xy[:, :, 1], 1)); length.append(np.nanstd(xy[:, :, 0], 1))
            push.append(np.nanmean(vx, 1) * attack); pace.append(np.nanmean(sp, 1))
    return np.column_stack([wid[0] - wid[1], length[0] - length[1], (push[0] + push[1]) / 2, pace[0] - pace[1]])
