"""hmm.keeper_hold.keeper_holds on small synthetic matches."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.keeper_hold import keeper_holds  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)


def match(end_ball):
    """30 s: A attacks the home goal at x=0 for 10 s, the ball reaches the home keeper (slot 0)
    at (3, 34) and is lost for 10 s (decoded D), then is seen again at end_ball."""
    n = 30 * FPS
    px = np.full((n, 4), np.nan)
    py = np.full((n, 4), np.nan)
    px[:, 0], py[:, 0] = 3.0, 34.0            # home keeper
    px[:, 1], py[:, 1] = 30.0, 20.0           # home defender
    px[:, 2], py[:, 2] = 25.0, 40.0           # away forward
    px[:, 3], py[:, 3] = 90.0, 34.0           # away keeper
    team = np.array([0, 0, 1, 1])
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    t = np.arange(10 * FPS)
    bx[t] = 25.0 - 22.0 * t / len(t)          # a shot towards the keeper
    by[t] = 34.0
    bx[20 * FPS:], by[20 * FPS:] = end_ball
    px[20 * FPS:, 0], py[20 * FPS:, 0] = end_ball[0] - 0.5, end_ball[1]
    labels = np.array(["A"] * (10 * FPS) + ["D"] * (10 * FPS) + ["H"] * (10 * FPS))
    return labels, px, py, team, bx, by


def test_keeper_hold_is_given_to_the_keepers_team():
    labels, px, py, team, bx, by = match((10.0, 34.0))   # thrown out from his area
    out, fired = keeper_holds(labels, px, py, team, bx, by, PITCH, FPS)
    assert len(fired) == 1
    assert np.all(out[10 * FPS:20 * FPS] == "H")
    assert np.all(out[:10 * FPS] == labels[:10 * FPS])


def test_goal_kick_from_the_goal_area_is_left_dead():
    labels, px, py, team, bx, by = match((4.0, 30.0))    # placed in the goal area
    out, fired = keeper_holds(labels, px, py, team, bx, by, PITCH, FPS)
    assert not fired
    assert np.all(out == labels)


def test_ball_seen_again_far_from_the_keeper_is_left_dead():
    labels, px, py, team, bx, by = match((0.5, 67.5))    # a corner
    px[20 * FPS:, 0], py[20 * FPS:, 0] = 3.0, 34.0
    out, fired = keeper_holds(labels, px, py, team, bx, by, PITCH, FPS)
    assert not fired


def test_clearance_lost_out_of_play_is_left_dead():
    """A home defender clears it, the ball is lost and away restart (A after the stoppage)."""
    labels, px, py, team, bx, by = match((10.0, 34.0))
    labels[20 * FPS:] = "A"
    bx[20 * FPS:], by[20 * FPS:] = np.nan, np.nan
    out, fired = keeper_holds(labels, px, py, team, bx, by, PITCH, FPS)
    assert not fired
    out, fired = keeper_holds(labels, px, py, team, bx, by, PITCH, FPS, blind_ok=True)
    assert not fired
