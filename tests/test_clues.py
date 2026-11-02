"""Turning one line of raw ATD into the gap clue."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import (  # noqa: E402
    BALL_MISSING, gap_category, line_distances, load_distances, read_distances)

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


def _write_atd(folder, lines):
    path = os.path.join(str(folder), "atd.txt")
    with open(path, "w") as handle:
        handle.writelines(lines)
    return path


def test_read_distances_counts_malformed_and_missing_lines(tmp_path):
    path = _write_atd(tmp_path, ["1:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n",
                                 "garbage\n",
                                 "3:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n"])
    ball_seen, _, d_away, stats = read_distances(path, TEAMS, 4, PITCH)
    assert ball_seen.tolist() == [True, False, False, False]
    assert np.isclose(d_away[0], 5.0)
    assert stats == (3, 1, 2)  # frames 1 and 3 (0-based) have no line


def test_read_distances_rejects_a_frame_outside_the_video(tmp_path):
    path = _write_atd(tmp_path, ["5:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n"])
    with pytest.raises(ValueError):
        read_distances(path, TEAMS, 4, PITCH)
    path = _write_atd(tmp_path, ["0:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n"])
    with pytest.raises(ValueError):
        read_distances(path, TEAMS, 4, PITCH)


def test_cache_is_reused_only_for_the_same_inputs(tmp_path):
    path = _write_atd(tmp_path, ["1:0.5,0.6,1.0;0.5,0.6,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n"])
    cache = os.path.join(str(tmp_path), "cache")
    first = load_distances(path, TEAMS, 1, PITCH, cache)
    again = load_distances(path, TEAMS, 1, PITCH, cache)
    assert np.allclose(first[1], again[1]) and first[3] == again[3]
    assert len(os.listdir(cache)) == 1

    swapped = load_distances(path, 1 - TEAMS, 1, PITCH, cache)  # home/away slot map changed
    assert len(os.listdir(cache)) == 2
    assert np.isclose(swapped[1][0], first[2][0])  # home and away distances trade places
