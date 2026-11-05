"""Turning raw ATD into the clues: the nearest-player gap, where the ball is, and how fast the
players move -- and which team a goalkeeper slot belongs to."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import (  # noqa: E402
    IN_PLAY, MAX_SPEED, MIN_PLAYERS, MISSING_AFTER_LINE, MISSING_AFTER_PLAY, MISSING_STALE, OUTSIDE, SPEED_LAG,
    KEEPER_ZONE, ball_is_usable, ball_place_category, gap_category, home_defends_low_x, keeper_slots, line_ball,
    line_positions, load_distances, median_player_speed, metres_outside, nearest_by_team,
    read_distances, speed_category, team_by_frame)

PITCH = (100.0, 50.0)
FPS = 25
TEAMS = np.array([0, 1, 0], dtype=np.int8)  # slots: home, away, home


def test_nearest_by_team_takes_the_nearest_tracked_slot_of_each_team():
    # Ball at (50 m, 25 m). Slot 0 home 10 m away along x, slot 1 away 5 m along y, slot 2 home 3 m.
    px = np.array([[60.0, 50.0, 50.0], [60.0, 50.0, np.nan], [60.0, 50.0, 50.0]])
    py = np.array([[25.0, 30.0, 22.0], [25.0, 30.0, np.nan], [25.0, 30.0, 22.0]])
    ball_x, ball_y = np.array([50.0, 50.0, np.nan]), np.array([25.0, 25.0, np.nan])
    d_home, d_away = nearest_by_team(px, py, ball_x, ball_y, np.tile(TEAMS, (3, 1)))
    assert np.allclose(d_home, [3.0, 10.0, np.nan], equal_nan=True)  # slot 2 untracked on frame 1
    assert np.allclose(d_away, [5.0, 5.0, np.nan], equal_nan=True)   # no ball on frame 2


def test_line_ball_scales_to_metres_and_is_nan_when_missing():
    assert np.allclose(line_ball("7:0.6,0.5,1.0;NaN,NaN,NaN:0.5,0.4,0.0\n", PITCH), (50.0, 20.0))
    assert np.isnan(line_ball("7:0.6,0.5,1.0;NaN,NaN,NaN:NaN,NaN,NaN\n", PITCH)).all()


def test_line_positions_scales_every_slot_to_metres_with_nan_for_the_untracked():
    frame, xs, ys = line_positions("7:0.6,0.5,1.0;NaN,NaN,NaN;0.5,0.44,0.0:NaN,NaN,NaN\n", 3, PITCH)
    assert frame == 6
    assert np.allclose(xs, [60.0, np.nan, 50.0], equal_nan=True)
    assert np.allclose(ys, [25.0, np.nan, 22.0], equal_nan=True)


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
    ball_x, _, _, d_away, _, stats = read_distances(path, TEAMS, 4, PITCH, FPS)
    assert (~np.isnan(ball_x)).tolist() == [True, False, False, False]
    assert np.isclose(d_away[0], 5.0)
    assert stats == (3, 1, 2, 0, 0)  # frames 1 and 3 (0-based) have no line; no keepers


def test_read_distances_rejects_a_frame_outside_the_video(tmp_path):
    path = _write_atd(tmp_path, ["5:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n"])
    with pytest.raises(ValueError):
        read_distances(path, TEAMS, 4, PITCH, FPS)
    path = _write_atd(tmp_path, ["0:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n"])
    with pytest.raises(ValueError):
        read_distances(path, TEAMS, 4, PITCH, FPS)


def test_cache_is_reused_only_for_the_same_inputs(tmp_path):
    path = _write_atd(tmp_path, ["1:0.5,0.6,1.0;0.5,0.6,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n"])
    cache = os.path.join(str(tmp_path), "cache")
    first = load_distances(path, TEAMS, 1, PITCH, FPS, cache)
    again = load_distances(path, TEAMS, 1, PITCH, FPS, cache)
    assert np.allclose(first[0], again[0]) and np.allclose(first[3], again[3])
    assert first[5] == again[5]
    assert len(os.listdir(cache)) == 1

    swapped = load_distances(path, 1 - TEAMS, 1, PITCH, FPS, cache)  # home/away slot map changed
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


def test_a_ball_fetched_far_behind_the_goal_line_stays_lost_at_the_line():
    nan = np.nan
    # Last seen on the goal line, then tracked 8 m behind it while fetched, then lost.
    assert _place([0.5, -8, -8, nan, nan], [25] * 3 + [nan] * 2, carry=4) == (
        [IN_PLAY] + [MISSING_AFTER_LINE] * 4)


def _walkers(n, speeds):
    t = np.arange(n, dtype=float)[:, None]
    px = np.hstack([t * v / FPS for v in speeds])
    return px, np.zeros_like(px)


def test_median_player_speed_is_over_a_centred_lag_and_ignores_identity_jumps():
    n = 3 * SPEED_LAG
    # Players at 1, 2 and 3 m/s along x, and one that jumps far (an identity swap).
    px, py = _walkers(n, [1.0, 2.0, 3.0] * MIN_PLAYERS + [2 * MAX_SPEED])
    speed = median_player_speed(px, py, FPS)
    behind, ahead = SPEED_LAG - SPEED_LAG // 2, SPEED_LAG // 2
    assert np.isnan(speed[:behind]).all() and np.isnan(speed[n - ahead:]).all()
    assert np.allclose(speed[behind:n - ahead], 2.0)


def test_a_speed_over_too_few_players_counts_as_unmeasured():
    px, py = _walkers(3 * SPEED_LAG, [1.0] * MIN_PLAYERS)
    px[2 * SPEED_LAG:, 0] = np.nan  # one player lost: MIN_PLAYERS - 1 left
    speed = median_player_speed(px, py, FPS)
    assert np.isfinite(speed[SPEED_LAG]) and np.isnan(speed[2 * SPEED_LAG:]).all()


def test_speed_category_bins_from_standing_to_running_and_skips_nan():
    cats = speed_category(np.array([0.0, 1.0, 1.7, 10.0, np.nan]))
    assert cats[0] == 0 and cats[3] == cats.max() and cats[-1] == -1
    assert 0 < cats[1] < cats[2] < cats[3]


def test_read_distances_measures_how_fast_the_players_move(tmp_path):
    # Slot 0 moves 0.1 of the 100 m pitch (10 m) every frame: 250 m/s, an identity jump, ignored.
    # The other slots move 0.0008 of 50 m (0.04 m) a frame: 1 m/s.
    n_slots = MIN_PLAYERS + 1
    teams = np.arange(n_slots, dtype=np.int8) % 2
    n = 3 * SPEED_LAG
    lines = ["{}:{:.4f},0.5,0;{}:0.5,0.5,0\n".format(
        f + 1, (0.1 * f) % 1, ";".join(["0.5,{:.4f},0".format(0.1 + 0.0008 * f)] * MIN_PLAYERS))
        for f in range(n)]
    *_, speed, _ = read_distances(_write_atd(tmp_path, lines), teams, n, PITCH, FPS)
    assert np.allclose(speed[SPEED_LAG:2 * SPEED_LAG], 1.0)


# Keepers. Slots: 0 home outfield, 1 away outfield, 2 home keeper, 3 away keeper. Home's outfield
# stands at x = 30 m, away's at 70 m, so home defends the x = 0 end.
KEEPER_TEAMS = np.array([0, 1, 0, 1], dtype=np.int8)
KEEPERS = (2, 3)


def _keeper_positions(home_keeper_x, away_keeper_x):
    n = len(home_keeper_x)
    return np.column_stack([np.full(n, 30.0), np.full(n, 70.0), home_keeper_x, away_keeper_x])


def test_the_side_a_team_defends_comes_from_its_outfield_slots():
    px = _keeper_positions([95.0] * 3, [5.0] * 3)  # keepers at the wrong ends do not decide it
    assert home_defends_low_x(px, KEEPER_TEAMS, KEEPERS)
    assert not home_defends_low_x(px[:, [1, 0, 2, 3]], KEEPER_TEAMS, KEEPERS)


def test_a_keeper_slot_in_the_other_teams_half_counts_for_that_team():
    # Frame 0: both keepers at their own ends. Frame 1: the away keeper's slot carries the home
    # keeper in the home goal. Frame 2: the home keeper's slot is untracked. Frame 3 is halftime.
    px = _keeper_positions([5.0, 8.0, np.nan, 95.0], [100.0, 6.0, 100.0, 6.0])
    team = team_by_frame(px, KEEPER_TEAMS, KEEPERS, [(0, 3)], 105.0)
    assert team[:, 3].tolist() == [1, 0, 1, 1]  # outside the halves the metadata's map holds
    assert team[:, 2].tolist() == [0, 0, 0, 0]
    assert (team[:, :2] == KEEPER_TEAMS[:2]).all()  # outfield slots never move


def test_the_keeper_rule_follows_the_teams_switching_ends_at_halftime():
    first = _keeper_positions([5.0, 5.0], [100.0, 100.0])
    second = first[:, [1, 0, 3, 2]].copy()  # everyone at the other end
    second[1, 3] = 90.0  # the away keeper's slot in the home goal, now at the high-x end
    team = team_by_frame(np.vstack([first, second]), KEEPER_TEAMS, KEEPERS, [(0, 2), (2, 4)], 105.0)
    assert team[:, 2].tolist() == [0, 0, 0, 0]
    assert team[:, 3].tolist() == [1, 1, 1, 0]


def test_read_distances_gives_the_ball_to_the_keeper_standing_in_his_teams_goal(tmp_path):
    # The ball sits 1 m from slot 3, the away keeper's slot, which stands in the home goal (x = 5 m
    # on a 100 m pitch where home defends x = 0). The nearest home slot is otherwise 25 m away.
    line = "{}:0.3,0.5,0;0.7,0.5,0;NaN,NaN,NaN;0.05,0.5,0:0.06,0.5,0\n"  # home keeper untracked
    path = _write_atd(tmp_path, [line.format(f + 1) for f in range(2)])
    _, _, d_home, d_away, _, stats = read_distances(path, KEEPER_TEAMS, 2, PITCH, FPS,
                                                    keepers=KEEPERS, halves=[(0, 1)])
    assert np.isclose(d_home[0], 1.0) and np.isclose(d_away[0], 64.0)  # inside the half
    assert np.isclose(d_home[1], 24.0) and np.isclose(d_away[1], 1.0)  # outside: metadata's map
    assert stats[3:] == (1, 1)  # one keeper frame moved, with the ball 1 m away


def test_keeper_slots_reads_the_goalkeeper_flag_from_the_metadata(tmp_path):
    def player(kind):
        return ("<Player teamId='t'><ProviderPlayerParameters><ProviderParameter><Name>"
                "position_type</Name><Value>{}</Value></ProviderParameter>"
                "</ProviderPlayerParameters></Player>").format(kind)
    path = os.path.join(str(tmp_path), "meta.xml")
    with open(path, "w") as handle:
        handle.write("<main><Players>{}</Players></main>".format(
            "".join(player(k) for k in ("Unknown", "Goalkeeper", "Unknown", "Goalkeeper"))))
    assert keeper_slots(path) == (1, 3)


def test_a_keeper_slot_away_from_both_goals_keeps_the_metadata_team():
    # The home keeper's slot sweeping just past the zone, and on halfway, and the away keeper's
    # slot carried onto an outfield player in the home half: none of them is moved.
    px = _keeper_positions([KEEPER_ZONE + 1, 52.5, 5.0], [100.0, 100.0, 40.0])
    team = team_by_frame(px, KEEPER_TEAMS, KEEPERS, [(0, 3)], 105.0)
    assert team[:, 2].tolist() == [0, 0, 0] and team[:, 3].tolist() == [1, 1, 1]


def test_keepers_must_be_one_per_team():
    px = _keeper_positions([5.0], [100.0])
    with pytest.raises(ValueError):
        team_by_frame(px, KEEPER_TEAMS, (0, 2), [(0, 1)], 105.0)  # two home slots
    assert (team_by_frame(px, KEEPER_TEAMS, (), [(0, 1)], 105.0) == KEEPER_TEAMS).all()


def test_halves_where_the_teams_do_not_switch_ends_are_an_error():
    px = np.vstack([_keeper_positions([5.0], [100.0])] * 2)
    with pytest.raises(ValueError):
        team_by_frame(px, KEEPER_TEAMS, KEEPERS, [(0, 1), (1, 2)], 105.0)
