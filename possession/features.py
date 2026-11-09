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


def spread(players, team):
    """Per frame, each team's mean distance from its own centre, in metres: (home, away)."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with a side untracked
        for t in (0, 1):
            xy = players[:, team == t, :2]
            out.append(np.nanmean(np.linalg.norm(xy - np.nanmean(xy, 1, keepdims=True), axis=-1), 1))
    return out
