"""Per-frame measurements from the tracking arrays (possession/load.py, external/build.py)."""

import warnings

import numpy as np

from load import PITCH  # noqa: F401  (external/score.py repoints this per match)


def nearest_by_team(players, ball, team):
    """Distance in metres from the ball to the nearest home and nearest away player, NaN where
    the ball or that whole side is missing."""
    d = np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        return np.nanmin(d[:, team == 0], 1), np.nanmin(d[:, team == 1], 1)


def own_goal(players, team, halves):
    """Per team, per half: +1 if its own goal is at x=0, -1 if at x=PITCH[0], read from which side
    of halfway its players stand on average."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [[1 if np.nanmean(players[a:b + 1, team == t, 0]) < PITCH[0] / 2 else -1 for a, b in halves]
                for t in (0, 1)]


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
