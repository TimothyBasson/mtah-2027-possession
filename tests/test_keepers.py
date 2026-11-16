"""possession/keepers.py: a keeper slot in the other team's goal changes team; frozen tracks hide."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "possession"), os.path.join(ROOT, "external")]
import keepers as KP  # noqa: E402


def _speed(xy, fps, lag=5):
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    v[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1) * fps / lag
    return v


def test_keeper_in_wrong_goal_moves_and_frozen_hides():
    n, L, fps = 200, 105.0, 25
    rng = np.random.default_rng(0)
    team = np.array([0, 0, 1, 1])          # slot 1 home keeper, slot 3 away keeper
    xy = np.zeros((n, 4, 2), np.float32)
    xy[:, 0] = [40, 30]; xy[:, 2] = [60, 30]   # home outfield deeper at low x
    xy[:, 1] = [5, 34]                     # home keeper in his goal
    xy[:, 3] = [100, 34]                   # away keeper in his goal
    xy += rng.normal(0, 0.1, xy.shape).astype(np.float32)
    xy[50:100, 3] = [4, 30] + rng.normal(0, 0.1, (50, 2))  # away keeper slot carried into the home goal
    xy[120:160, 3] = [100, 34]             # away keeper slot frozen for 1.6 s
    players = np.concatenate([xy, _speed(xy, fps)[..., None]], -1)
    out, team2 = KP.fix(players, team, [(0, n - 1)], (1, 3), L, fps, _speed)
    assert team2.tolist() == [0, 0, 1, 1, 0, 1]
    assert np.isnan(out[50:100, 3, 0]).all() and np.isfinite(out[50:100, 4, 0]).all()   # moved to home's extra slot
    assert np.isnan(out[120:160, 3, 0]).all()                                          # frozen: unseen
    assert np.isfinite(out[:50, 3, 0]).all() and np.isnan(out[:, 5, 0]).all()
    assert np.isfinite(out[:, 1, 0]).all()
