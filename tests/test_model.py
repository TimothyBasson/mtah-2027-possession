"""The HMM core on sequences small enough to check by hand."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.model import (  # noqa: E402
    H, A, D, baum_welch, build_log_emit, forward_backward, rate_log_trans, sticky_log_trans, viterbi)


def test_worked_trellis_stays_home():
    # Costs per frame for H/A/D. Frame 1 prefers A by 1.2, but leaving H and coming back costs
    # 3 + 3, so the cheapest path never leaves H.
    costs = np.array([[0.3, 1.5, 2.0],
                      [1.5, 0.3, 2.0],
                      [0.3, 1.5, 2.0],
                      [0.0, 0.0, 0.0]])
    log_trans = -np.where(np.eye(3, dtype=bool), 0.0, 3.0)
    path = viterbi(-costs, log_trans, np.zeros(3))
    assert path.tolist() == [H, H, H, H]


def test_cheap_switch_follows_the_evidence():
    costs = np.array([[0.3, 1.5, 2.0],
                      [1.5, 0.3, 2.0],
                      [0.3, 1.5, 2.0]])
    log_trans = -np.where(np.eye(3, dtype=bool), 0.0, 0.5)
    assert viterbi(-costs, log_trans, np.zeros(3)).tolist() == [H, A, H]


def test_start_prior_decides_a_tie():
    log_trans = np.log(sticky_log_trans_probs(0.01))
    path = viterbi(np.zeros((5, 3)), log_trans, np.log([0.1, 0.1, 0.8]))
    assert path.tolist() == [D] * 5


def sticky_log_trans_probs(p):
    return np.exp(sticky_log_trans(p))


def test_sticky_rows_are_distributions():
    probs = sticky_log_trans_probs(0.03)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(np.diag(probs), 0.97)
    assert np.allclose(probs[0, 1], 0.015)


def test_rate_transitions_give_the_requested_durations():
    probs = np.exp(rate_log_trans(possession_frames=300, stoppage_frames=500, to_dead_share=0.4))
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Expected stay of a geometric run is 1 / (1 - p_stay).
    assert 1.0 / (1.0 - probs[H, H]) == pytest.approx(300)
    assert 1.0 / (1.0 - probs[D, D]) == pytest.approx(500)
    assert probs[H, D] / (1.0 - probs[H, H]) == pytest.approx(0.4)
    assert probs[D, H] == pytest.approx(probs[D, A])
    # A mirrors H.
    assert probs[A, A] == pytest.approx(probs[H, H])
    assert probs[A, D] == pytest.approx(probs[H, D])
    assert probs[A, H] == pytest.approx(probs[H, A])


def test_build_log_emit_sums_weighted_clues_and_skips_missing():
    table_1 = np.log(np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]))  # state x category
    table_2 = np.log(np.array([[0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]))
    obs_1 = np.array([0, 1, -1])
    obs_2 = np.array([1, -1, -1])
    out = build_log_emit([(obs_1, table_1, 1.0), (obs_2, table_2, 0.5)])
    assert out.shape == (3, 3)
    assert np.allclose(out[0], table_1[:, 0] + 0.5 * table_2[:, 1])
    assert np.allclose(out[1], table_1[:, 1])
    assert np.allclose(out[2], 0.0)  # nothing observed adds nothing


def test_forward_backward_posteriors_are_distributions_and_agree_with_brute_force():
    rng = np.random.default_rng(0)
    log_emit = np.log(rng.uniform(0.1, 1.0, size=(4, 3)))
    log_trans = np.log(sticky_log_trans_probs(0.2))
    log_start = np.log([0.2, 0.3, 0.5])
    post, xi_sum, loglik = forward_backward(log_emit, log_trans, log_start)
    assert np.allclose(post.sum(axis=1), 1.0)
    assert np.allclose(xi_sum.sum(), 3.0)  # T - 1 transitions

    # Brute force over all 3^4 paths.
    import itertools
    total, marg = 0.0, np.zeros((4, 3))
    for path in itertools.product(range(3), repeat=4):
        lp = log_start[path[0]] + log_emit[0, path[0]]
        for t in range(1, 4):
            lp += log_trans[path[t - 1], path[t]] + log_emit[t, path[t]]
        total += np.exp(lp)
        for t, s in enumerate(path):
            marg[t, s] += np.exp(lp)
    assert loglik == pytest.approx(np.log(total))
    assert np.allclose(post, marg / total)


def test_baum_welch_never_lowers_the_likelihood_and_recovers_a_clear_chain():
    rng = np.random.default_rng(1)
    true_tables = [np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                   np.array([[0.7, 0.3], [0.5, 0.5], [0.1, 0.9]])]
    states = np.repeat(rng.integers(0, 3, size=40), 50)
    clues = []
    for k, true_table in enumerate(true_tables):
        obs = np.array([rng.choice(true_table.shape[1], p=true_table[s]) for s in states])
        obs[k::7] = -1  # some frames unobserved, different ones per clue
        start = np.full(true_table.shape, 1.0 / true_table.shape[1])
        start[np.arange(3), np.arange(3) % true_table.shape[1]] += 0.2
        clues.append((obs, np.log(start / start.sum(axis=1, keepdims=True))))

    tables, _, history = baum_welch(clues, sticky_log_trans(0.05), np.log(np.full(3, 1 / 3)),
                                    n_iter=15, segments=[(0, 1000), (1000, 2000)])
    assert all(b >= a - 1e-6 for a, b in zip(history, history[1:]))
    for table, true_table in zip(tables, true_tables):  # ~1.7k draws: sampling noise
        assert np.allclose(np.exp(table), true_table, atol=0.08)
