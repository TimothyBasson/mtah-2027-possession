"""The football wiring in hmm/decode.py: frames before a kickoff are dead, and so is a ball that
went out of play, or that went missing while the players walked; a ball in flight belongs to the
team that kicked it; a player on the ball counts for more than one who is merely nearer; and what
a missing ball means is read off the match itself."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.ball import N_PHASES  # noqa: E402
from hmm.clues import N_WIDTH_BINS  # noqa: E402
from hmm.clues import (  # noqa: E402
    IN_PLAY, MISSING_AFTER_PLAY, MISSING_STALE, N_DISTANCE_BINS, N_GAP_BINS, ball_place_category,
    mirror_gap_distance, speed_category)
from hmm.decode import (  # noqa: E402
    CARRY_SECONDS, FPS, KICKOFFS, PLACE_FLOOR, START, dead_runs, decode, fit_place, flight_table,
    gap_distance_table, gap_table, graded_segments, pin_dead_before_kickoff, place_placebo,
    place_table, speed_table, transitions, width_table, WEIGHT, WIDTH_WEIGHT)

HOME_MUCH_CLOSER = N_GAP_BINS - 1
PITCH = (105.0, 68.0)


def _home_dominates(n):
    return [(np.full(n, HOME_MUCH_CLOSER, dtype=np.int8), gap_table()),
            (np.full(n, IN_PLAY, dtype=np.int8), place_table())]


def test_frames_before_a_kickoff_are_dead_even_against_strong_evidence():
    labels = decode(_home_dominates(3000), transitions(), np.log(START),
                    segments=[(100, 2900)], kickoffs=(1000,))
    assert set(labels[:100]) == {"X"} and set(labels[2900:]) == {"X"}
    assert set(labels[100:1000]) == {"D"}
    assert set(labels[1000:2900]) == {"H"}


def test_a_kickoff_at_the_start_of_a_half_pins_nothing():
    labels = decode(_home_dominates(500), transitions(), np.log(START),
                    segments=[(0, 500)], kickoffs=(0,))
    assert set(labels) == {"H"}


def test_published_kickoffs_pin_exactly_the_second_half_warm_up():
    # FAQ.md: the second half is graded from 110,808 but kicks off at 113,742, and those 2,934
    # frames are all D. The first half is graded from its kickoff, so nothing is pinned there.
    (lo1, hi1), (lo2, hi2) = graded_segments()
    assert KICKOFFS == (lo1, 113742)
    log_emit = np.zeros((hi2 - lo2, 3))
    pin_dead_before_kickoff(log_emit, lo2, hi2, KICKOFFS)
    pinned = np.flatnonzero(log_emit[:, 0] < 0)
    assert len(pinned) == 2934 and pinned[-1] + lo2 == 113741


def _gap_then_missing(last_y, speed_while_missing=1.7):
    """Home on the ball for 40 s, the ball last seen at (50, last_y), missing for 10 s, home again.
    The players run at 2.2 m/s while home has the ball, and at speed_while_missing during the gap."""
    live, gone = 1000, 10 * FPS
    n = 2 * live + gone
    x = np.full(n, 50.0)
    y = np.full(n, 34.0)
    y[live - 1] = last_y
    x[live:live + gone] = y[live:live + gone] = np.nan
    place = ball_place_category(x, y, PITCH, CARRY_SECONDS * FPS)
    gap = np.full(n, HOME_MUCH_CLOSER, dtype=np.int8)
    gap[live:live + gone] = -1
    speed = np.full(n, 2.2)
    speed[live:live + gone] = speed_while_missing
    labels = decode([(gap, gap_table()), (place, place_table()),
                     (speed_category(speed), speed_table())], transitions(), np.log(START),
                    segments=[(0, n)], kickoffs=())
    return labels[live:live + gone]


def test_a_ball_lost_at_the_touchline_is_a_stoppage():
    assert np.mean(_gap_then_missing(last_y=0.5) == "D") > 0.9


def test_a_ball_lost_in_open_play_stays_with_the_team():
    assert set(_gap_then_missing(last_y=34.0)) == {"H"}


def test_a_ball_lost_in_open_play_while_the_players_walk_is_a_stoppage():
    assert np.mean(_gap_then_missing(last_y=34.0, speed_while_missing=0.5) == "D") > 0.8


def test_players_slowing_down_while_the_ball_is_seen_is_not_enough_for_a_stoppage():
    # Walking pace (1.3 m/s) happens in live play too, so with the ball seen and home close to it,
    # it stays H.
    n = 2000
    speed = np.full(n, 2.2)
    speed[800:1200] = 1.3
    clues = _home_dominates(n) + [(speed_category(speed), speed_table())]
    labels = decode(clues, transitions(), np.log(START), segments=[(0, n)], kickoffs=())
    assert set(labels) == {"H"}


def test_flight_table_mirrors_the_teams_and_is_neutral_for_dead():
    table = np.exp(flight_table())
    assert np.allclose(table.sum(axis=1), 1.0)
    assert np.allclose(table[1, N_PHASES:2 * N_PHASES], table[0, :N_PHASES])
    assert np.allclose(table[2], (table[0] + table[1]) / 2)


def _clearance(with_flight):
    """Away on the ball for 20 s, then clears it: for 3 s it flies over the home team, who are
    much closer to it, and then home has it for 20 s."""
    before, flying, after = 20 * FPS, 3 * FPS, 20 * FPS
    n = before + flying + after
    gap = np.full(n, HOME_MUCH_CLOSER, dtype=np.int8)
    gap[:before] = 0  # away much closer
    clues = [(gap, gap_table()), (np.full(n, IN_PLAY, dtype=np.int8), place_table()),
             (speed_category(np.full(n, 2.2)), speed_table())]
    if with_flight:
        flight = np.full(n, -1, dtype=np.int8)
        seconds = np.arange(flying) / FPS
        flight[before:before + flying] = N_PHASES + np.digitize(seconds, [0.5, 1.5])  # away kicked
        clues.append((flight, flight_table()))
    labels = decode(clues, transitions(), np.log(START), segments=[(0, n)], kickoffs=())
    return labels[before:before + flying], labels[before + flying + 2 * FPS:]


def test_a_clearance_stays_with_the_kicker_while_it_flies():
    flying, landed = _clearance(with_flight=True)
    assert np.mean(flying == "A") > 0.9 and set(landed) == {"H"}
    flying, _ = _clearance(with_flight=False)  # without the clue, home takes it at the kick
    assert np.mean(flying == "H") > 0.5


def test_gap_distance_table_mirrors_the_teams_and_dead_favours_neither():
    table = np.exp(gap_distance_table())
    assert np.allclose(table.sum(axis=1), 1.0)
    assert np.allclose(table[1], mirror_gap_distance(table[0]))
    assert np.allclose(table[2], mirror_gap_distance(table[2]))


def _away_nearer_for(seconds, distance_bin):
    """Home on the ball for 20 s, then away much the nearer for `seconds` with their nearest player
    in distance_bin, then home on the ball again. Returns the labels of the middle stretch."""
    before, middle = 20 * FPS, int(seconds * FPS)
    n = 2 * before + middle
    home_on_ball = (N_GAP_BINS - 1) * N_DISTANCE_BINS
    gap = np.full(n, home_on_ball, dtype=np.int8)
    gap[before:before + middle] = distance_bin  # gap bin 0: away much closer
    labels = decode([(gap, gap_distance_table()),
                     (np.full(n, IN_PLAY, dtype=np.int8), place_table()),
                     (speed_category(np.full(n, 2.2)), speed_table())], transitions(),
                    np.log(START), segments=[(0, n)], kickoffs=())
    return labels[before:before + middle]


def test_a_short_spell_with_a_player_on_the_ball_is_a_possession():
    assert np.mean(_away_nearer_for(3, distance_bin=0) == "A") > 0.9


def test_being_nearer_a_ball_nobody_is_near_is_not_a_possession():
    # the same 3 s and the same gap, but the nearest away player is over 5 m from the ball
    assert set(_away_nearer_for(3, distance_bin=N_DISTANCE_BINS - 1)) == {"H"}


def _match_where_the_ball(goes_missing_when_dead, speeds=(2.6, 0.4), gap_seconds=1):
    """60 s of play (home on the ball, players running) then 60 s of stoppage (nobody near the
    ball, players standing), three times over. speeds are the players' in play and in a stoppage;
    a ball lost at random is missing every other gap_seconds. Returns (place, the other clues)."""
    block = 60 * FPS
    live = np.tile(np.r_[np.ones(block, bool), np.zeros(block, bool)], 3)
    n = len(live)
    home_on_ball, nobody_near = (N_GAP_BINS - 1) * N_DISTANCE_BINS, 2 * N_DISTANCE_BINS + 3
    gap = np.where(live, home_on_ball, nobody_near).astype(np.int8)
    speed = speed_category(np.where(live, *speeds))
    if goes_missing_when_dead:
        place = np.where(live, IN_PLAY, MISSING_AFTER_PLAY).astype(np.int8)
    else:  # live or dead alike
        place = np.where(np.arange(n) // (gap_seconds * FPS) % 2 == 0, IN_PLAY,
                         MISSING_AFTER_PLAY).astype(np.int8)
    gap[place != IN_PLAY] = -1  # no ball, no gap: as load_observations has it
    return place, [(gap, gap_distance_table()), (speed, speed_table())]


def test_fit_place_learns_that_this_tracking_loses_the_ball_in_stoppages():
    place, others = _match_where_the_ball(goes_missing_when_dead=True)
    live, dead = fit_place(place, others, transitions(), np.log(START), [(0, len(place))],
                           kickoffs=())
    assert live[IN_PLAY] > 0.85 and dead[MISSING_AFTER_PLAY] > 0.85
    assert np.isclose(live.sum(), 1.0) and np.isclose(dead.sum(), 1.0)


def test_fit_place_learns_nothing_from_a_ball_lost_at_random():
    place, others = _match_where_the_ball(goes_missing_when_dead=False)
    live, dead = fit_place(place, others, transitions(), np.log(START), [(0, len(place))],
                           kickoffs=())
    assert abs(live[IN_PLAY] - dead[IN_PLAY]) < 0.05


def test_fit_place_leans_only_a_little_towards_missing_means_dead_on_a_ball_lost_at_random():
    # The gap clue pushes a frame with the ball in view towards live, and a frame without it has
    # only the players' speed. With speeds that separate play from a stoppage less sharply (a slow
    # run against a walk) and 4 s gaps, the live row comes out a few points more "seen" than the
    # dead row, where the truth is equal. This bounds that bias; fit_place's docstring has the
    # worst case and what it comes to on BAR-ATH.
    place, others = _match_where_the_ball(goes_missing_when_dead=False, speeds=(1.7, 1.0),
                                          gap_seconds=4)
    live, dead = fit_place(place, others, transitions(), np.log(START), [(0, len(place))],
                           kickoffs=())
    assert 0 < live[IN_PLAY] - dead[IN_PLAY] < 0.1


def test_place_placebo_finds_no_link_once_the_ball_is_shifted_against_play():
    # the ball is missing exactly in the stoppages: fitted, the two rows are far apart; shifted by
    # half a play-stoppage cycle against the other clues, they are not
    place, others = _match_where_the_ball(goes_missing_when_dead=True)
    (gap, _), (speed, _) = others
    live, dead = fit_place(place, others, transitions(), np.log(START), [(0, len(place))],
                           kickoffs=())
    assert live[IN_PLAY] - dead[IN_PLAY] > 0.8
    flight = np.full(len(place), -1, dtype=np.int8)
    [(_, seen_live, seen_dead)] = place_placebo(gap, place, speed, flight, transitions(),
                                                np.log(START), [(0, len(place))], kickoffs=(),
                                                minutes=(0.5,))
    assert abs(seen_live - seen_dead) < 0.2


def test_fit_place_never_lets_one_category_rule_a_state_out():
    # a stale ball is never seen in this match, live or dead, and still keeps the floor share
    place, others = _match_where_the_ball(goes_missing_when_dead=True)
    live, dead = fit_place(place, others, transitions(), np.log(START), [(0, len(place))],
                           kickoffs=())
    assert live[MISSING_STALE] >= PLACE_FLOOR * 0.9 and dead[MISSING_STALE] >= PLACE_FLOOR * 0.9
    assert live.min() > 0 and dead.min() > 0


def test_fit_place_counts_the_frames_before_a_kickoff_as_dead():
    # the players warm up at a run with the ball missing: without the kickoff that reads as live
    n, kickoff = 120 * FPS, 60 * FPS
    place = np.where(np.arange(n) < kickoff, MISSING_STALE, IN_PLAY).astype(np.int8)
    gap = np.full(n, (N_GAP_BINS - 1) * N_DISTANCE_BINS, dtype=np.int8)
    gap[:kickoff] = -1
    others = [(gap, gap_distance_table()), (speed_category(np.full(n, 2.6)), speed_table())]
    live, dead = fit_place(place, others, transitions(), np.log(START), [(0, n)],
                           kickoffs=(kickoff,))
    assert dead[MISSING_STALE] > 0.85 and live[MISSING_STALE] < 2 * PLACE_FLOOR
    live, _ = fit_place(place, others, transitions(), np.log(START), [(0, n)], kickoffs=())
    assert live[MISSING_STALE] > 0.3


def test_dead_runs_flags_a_stoppage_with_no_ball_and_the_same_team_either_side():
    #         H x100 | D x50 (ball missing from 25 frames before it) | H x100 | D x50 seen | A x100
    labels = np.array(list("H" * 100 + "D" * 50 + "H" * 100 + "D" * 50 + "A" * 100))
    place = np.full(len(labels), IN_PLAY, dtype=np.int8)
    place[75:150] = MISSING_AFTER_PLAY
    runs = dead_runs(labels, place, [(0, len(labels))])
    assert runs["start"].tolist() == [100, 250] and runs["stop"].tolist() == [150, 300]
    assert runs["same_team"].tolist() == [True, False]
    assert runs["blind"].tolist() == [True, False]
    assert np.isclose(runs["into_missing"][0], 25 / FPS) and np.isnan(runs["into_missing"][1])


def test_width_table_mirrors_the_teams_and_keeps_dead_neutral():
    table = np.exp(width_table())
    assert table.shape == (3, N_WIDTH_BINS) and np.allclose(table.sum(axis=1), 1.0)
    assert np.allclose(table[1], table[0][::-1])       # away is home with the bins reversed
    assert np.allclose(table[2], table[2][::-1])       # a stoppage favours neither team
    assert table[0, -1] > 5 * table[0, 0]              # home much wider says home has the ball


def test_width_names_the_team_while_the_ball_is_missing():
    # 20 s with no ball at all: the gap, place and flight clues are silent or neutral. Home is
    # the wider team for the first half of it, away for the second.
    n = 500
    silent = np.full(n, -1, dtype=np.int8)
    running = (np.full(n, 6, dtype=np.int8), speed_table())    # players at full pace: live
    width = np.r_[np.full(n // 2, N_WIDTH_BINS - 1), np.zeros(n // 2)].astype(np.int8)
    labels = decode([(silent, gap_distance_table()), running], transitions(), np.log(START),
                    segments=[(0, n)], kickoffs=(), extra=[(width, width_table(), WIDTH_WEIGHT)])
    assert (labels[50:n // 2 - 50] == "H").all() and (labels[n // 2 + 50:n - 50] == "A").all()


def test_width_does_not_override_a_player_on_the_ball():
    # Away is the wider team throughout, but a home player has the ball at his feet.
    n = 500
    home_on_ball = (np.full(n, (N_GAP_BINS - 1) * N_DISTANCE_BINS, dtype=np.int8),
                    gap_distance_table())
    in_play = (np.full(n, IN_PLAY, dtype=np.int8), place_table())
    width = np.zeros(n, dtype=np.int8)
    labels = decode([home_on_ball, in_play], transitions(), np.log(START), segments=[(0, n)],
                    kickoffs=(), extra=[(width, width_table(), WIDTH_WEIGHT)])
    assert WIDTH_WEIGHT < WEIGHT and (labels[25:] == "H").all()


def test_a_brief_possession_between_two_spells_of_the_other_team_is_folded():
    from hmm.decode import fold_brief_possessions
    from hmm.model import A, D, H
    path = np.array([H] * 10 + [A] * 3 + [H] * 10 + [A] * 3 + [D] * 5 + [H] * 2 + [A] * 9)
    out = fold_brief_possessions(path, 5)
    assert list(out[:23]) == [H] * 23            # H A H: the touch goes back to H
    assert list(out[23:26]) == [A] * 3           # next to a stoppage: left alone
    assert list(out[31:33]) == [H] * 2           # after a stoppage: left alone
    assert np.array_equal(fold_brief_possessions(path, 0), path)


def test_folding_repeats_until_nothing_brief_is_left():
    from hmm.decode import fold_brief_possessions
    from hmm.model import A, H
    path = np.array([H] * 8 + [A] * 2 + [H] * 3 + [A] * 8)
    # the 2-frame A run folds into H, which makes one 13-frame H run that stays
    assert list(fold_brief_possessions(path, 4)) == [H] * 13 + [A] * 8
