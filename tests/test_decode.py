"""The football wiring in hmm/decode.py: frames before a kickoff are dead."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import N_GAP_BINS  # noqa: E402
from hmm.decode import (  # noqa: E402
    KICKOFFS, START, decode, emission_table, graded_segments, pin_dead_before_kickoff, transitions)


def test_frames_before_a_kickoff_are_dead_even_against_strong_evidence():
    home_much_closer = N_GAP_BINS - 1
    cats = np.full(3000, home_much_closer, dtype=np.int8)
    labels = decode(cats, emission_table(), transitions(), np.log(START),
                    segments=[(100, 2900)], kickoffs=(1000,))
    assert set(labels[:100]) == {"X"} and set(labels[2900:]) == {"X"}
    assert set(labels[100:1000]) == {"D"}
    assert set(labels[1000:2900]) == {"H"}


def test_a_kickoff_at_the_start_of_a_half_pins_nothing():
    cats = np.full(500, N_GAP_BINS - 1, dtype=np.int8)
    labels = decode(cats, emission_table(), transitions(), np.log(START),
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
