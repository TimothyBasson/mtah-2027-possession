"""Turning raw ATD into the clues: the nearest-player gap and where the ball is."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import (  # noqa: E402
    IN_PLAY, MISSING_AFTER_LINE, MISSING_AFTER_PLAY, MISSING_STALE, OUTSIDE, ball_is_usable,
    ball_place_category, gap_category, line_distances, load_distances, metres_outside,
    read_distances)

PITCH = (100.0, 50.0)
TEAMS = np.array([0, 1, 0], dtype=np.int8)  # slots: home, away, home


def test_line_distances_scales_to_metres_and_takes_the_nearest_of_each_team():
    # Ball at (50 m, 25 m). Slot 0 home 10 m away along x, slot 1 away 5 m along y, slot 2 home 3 m.
    line = "7:0.6,0.5,1.0;0.5,0.6,2.0;0.5,0.44,0.0:0.5,0.5,0.0\n"
    frame, ball_x, ball_y, d_home, d_away = line_distances(line, TEAMS, PITCH)
    assert frame == 6  # frame_id counts from 1
    assert np.isclose(ball_x, 50.0) and np.isclose(ball_y, 25.0)
    assert np.isclose(d_home, 3.0)
    assert np.isclose(d_away, 5.0)


def test_line_distances_handles_missing_ball_and_missing_players():
    _, ball_x, ball_y, d_home, d_away = line_distances(
        "1:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n", TEAMS, PITCH)
    assert np.isnan(ball_x) and np.isnan(ball_y) and np.isnan(d_home) and np.isnan(d_away)

    _, ball_x, _, d_home, d_away = line_distances(
        "1:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n", TEAMS, PITCH)
    assert not np.isnan(ball_x) and np.isnan(d_home) and np.isclose(d_away, 5.0)


def test_gap_category_bins_away_minus_home_and_skips_frames_without_a_usable_ball():
    usable = np.array([True, True, True, True, True, True, False, True])
    d_home = np.array([9.0, 2.0, 2.0, 2.0, 2.0, 2.0, np.nan, np.nan])
    d_away = np.array([2.0, 0.0, 2.5, 4.5, 9.0, 5.0, np.nan, 2.0])
    #       gap:      -7    -2   0.5  2.5   7    3 (edge, inclusive)
    cats = gap_category(usable, d_home, d_away)
    assert cats.tolist() == [0, 1, 2, 3, 4, 3, -1, -1]


def _write_atd(folder, lines):
    path = os.path.join(str(folder), "atd.txt")
    with open(path, "w") as handle:
        handle.writelines(lines)
    return path


def test_read_distances_counts_malformed_and_missing_lines(tmp_path):
    path = _write_atd(tmp_path, ["1:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n",
                                 "garbage\n",
                                 "3:NaN,NaN,NaN;0.5,0.6,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n"])
    ball_x, _, _, d_away, stats = read_distances(path, TEAMS, 4, PITCH)
    assert (~np.isnan(ball_x)).tolist() == [True, False, False, False]
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
    assert np.allclose(first[0], again[0]) and np.allclose(first[3], again[3])
    assert first[4] == again[4]
    assert len(os.listdir(cache)) == 1

    swapped = load_distances(path, 1 - TEAMS, 1, PITCH, cache)  # home/away slot map changed
    assert len(os.listdir(cache)) == 2
    assert np.isclose(swapped[2][0], first[3][0])  # home and away distances trade places


def test_metres_outside_is_signed_distance_to_the_nearest_line():
    # Centre spot, 1 m inside a goal line, 2 m past it, 3 m past a touchline, and 3 m past the
    # far goal line while 1 m inside a touchline -- outside wins.
    x = np.array([50.0, 1.0, -2.0, 50.0, 103.0])
    y = np.array([25.0, 25.0, 25.0, 53.0, 49.0])
    assert np.allclose(metres_outside(x, y, PITCH), [-25.0, -1.0, 2.0, 3.0, 3.0])


def test_a_ball_far_outside_is_not_the_match_ball():
    x = np.array([50.0, -3.0, -8.0, np.nan])
    y = np.full(4, 25.0)
    assert ball_is_usable(x, y, PITCH).tolist() == [True, True, False, False]


def _place(x, y, carry=3):
    return ball_place_category(np.array(x, float), np.array(y, float), PITCH, carry).tolist()


def test_ball_place_marks_a_ball_seen_clearly_outside():
    #        in play  0.5 m out (on the line)  2 m out
    assert _place([50.0, -0.5, -2.0], [25.0] * 3) == [IN_PLAY, IN_PLAY, OUTSIDE]


def test_a_missing_ball_carries_where_it_was_last_seen_for_a_while():
    nan = np.nan
    # Last seen 1 m inside the touchline, then missing for 5 frames with a carry of 3.
    assert _place([50, 50] + [nan] * 5, [25, 1] + [nan] * 5) == (
        [IN_PLAY, IN_PLAY] + [MISSING_AFTER_LINE] * 3 + [MISSING_STALE] * 2)
    # Last seen in midfield.
    assert _place([50] + [nan] * 4, [25] + [nan] * 4) == (
        [IN_PLAY] + [MISSING_AFTER_PLAY] * 3 + [MISSING_STALE])
    # Last seen outside counts as at the line too.
    assert _place([-2.0, nan], [25, nan]) == [OUTSIDE, MISSING_AFTER_LINE]


def test_a_missing_ball_with_no_sighting_before_it_is_stale():
    nan = np.nan
    assert _place([nan, nan, 50], [nan, nan, 25]) == [MISSING_STALE, MISSING_STALE, IN_PLAY]


def test_a_far_outside_sighting_counts_as_missing_and_does_not_reset_the_carry():
    nan = np.nan
    # Midfield, then a "ball" 10 m outside (a spare ball), then missing: all carried from midfield.
    assert _place([50, -10, nan, nan, nan], [25, 25, nan, nan, nan]) == (
        [IN_PLAY] + [MISSING_AFTER_PLAY] * 3 + [MISSING_STALE])
