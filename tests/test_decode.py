"""The football wiring in hmm/decode.py: frames before a kickoff are dead, and so is a ball that
went out of play."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import IN_PLAY, N_GAP_BINS, ball_place_category  # noqa: E402
from hmm.decode import (  # noqa: E402
    CARRY_SECONDS, FPS, KICKOFFS, START, decode, gap_table, graded_segments,
    pin_dead_before_kickoff, place_table, transitions)

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


def _gap_then_missing(last_y):
    """Home on the ball for 40 s, the ball last seen at (50, last_y), missing for 10 s, home again."""
    live, gone = 1000, 10 * FPS
    n = 2 * live + gone
    x = np.full(n, 50.0)
    y = np.full(n, 34.0)
    y[live - 1] = last_y
    x[live:live + gone] = y[live:live + gone] = np.nan
    place = ball_place_category(x, y, PITCH, CARRY_SECONDS * FPS)
    gap = np.full(n, HOME_MUCH_CLOSER, dtype=np.int8)
    gap[live:live + gone] = -1
    labels = decode([(gap, gap_table()), (place, place_table())], transitions(), np.log(START),
                    segments=[(0, n)], kickoffs=())
    return labels[live:live + gone]


def test_a_ball_lost_at_the_touchline_is_a_stoppage():
    assert np.mean(_gap_then_missing(last_y=0.5) == "D") > 0.9


def test_a_ball_lost_in_open_play_stays_with_the_team():
    assert set(_gap_then_missing(last_y=34.0)) == {"H"}
