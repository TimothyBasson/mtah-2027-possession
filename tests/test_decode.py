"""The football wiring in hmm/decode.py: frames before a kickoff are dead."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.clues import N_GAP_BINS  # noqa: E402
from hmm.decode import START, decode, emission_table, transitions  # noqa: E402


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
