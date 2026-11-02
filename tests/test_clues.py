"""Turning one line of raw ATD into the gap clue."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import BALL_MISSING, gap_category, line_distances  # noqa: E402

PITCH = (100.0, 50.0)
TEAMS = np.array([0, 1, 0], dtype=np.int8)  # slots: home, away, home


def test_line_distances_scales_to_metres_and_takes_the_nearest_of_each_team():
    # Ball at (50 m, 25 m). Slot 0 home 10 m away along x, slot 1 away 5 m along y, slot 2 home 3 m.
    line = "7:0.6,0.5,1.0;0.5,0.6,2.0;0.5,0.44,0.0:0.5,0.5,0.0\n"
    frame, ball_seen, d_home, d_away = line_distances(line, TEAMS, PITCH)
    assert frame == 6  # frame_id counts from 1
    assert ball_seen
    assert np.isclose(d_home, 3.0)
    assert np.isclose(d_away, 5.0)


def test_line_distances_handles_missing_ball_and_missing_players():
    frame, ball_seen, d_home, d_away = line_distances(
        "1:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n", TEAMS, PITCH)
    assert not ball_seen and np.isnan(d_home) and np.isnan(d_away)

    _, ball_seen, d_home, d_away = line_distances(
        "1:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n", TEAMS, PITCH)
    assert ball_seen and np.isnan(d_home) and np.isclose(d_away, 5.0)


def test_gap_category_bins_away_minus_home():
    ball_seen = np.array([True, True, True, True, True, True, False, True])
    d_home = np.array([9.0, 2.0, 2.0, 2.0, 2.0, 2.0, np.nan, np.nan])
    d_away = np.array([2.0, 0.0, 2.5, 4.5, 9.0, 5.0, np.nan, 2.0])
    #       gap:      -7    -2   0.5  2.5   7    3 (edge, inclusive)
    cats = gap_category(ball_seen, d_home, d_away)
    assert cats.tolist() == [0, 1, 2, 3, 4, 3, BALL_MISSING, -1]
