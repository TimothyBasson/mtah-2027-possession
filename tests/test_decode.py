"""The football wiring in hmm/decode.py: frames before a kickoff are dead, and so is a ball that
went out of play, or that went missing while the players walked; and a ball in flight belongs to
the team that kicked it."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.ball import N_PHASES  # noqa: E402
from hmm.clues import IN_PLAY, N_GAP_BINS, ball_place_category, speed_category  # noqa: E402
from hmm.decode import (  # noqa: E402
    CARRY_SECONDS, FPS, KICKOFFS, START, decode, flight_table, gap_table, graded_segments,
    pin_dead_before_kickoff, place_table, speed_table, transitions)

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
