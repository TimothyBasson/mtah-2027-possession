import numpy as np
import pytest

from hmm import trees
from hmm.control import _sided_means, _sided_sums, _windowed_mean, features
from hmm.model import viterbi


def test_windowed_mean_skips_nan():
    x = np.array([1.0, np.nan, 3.0, np.nan, np.nan])
    out = _windowed_mean(x, 1)
    assert out[0] == 1.0 and out[1] == 2.0 and out[2] == 3.0 and out[3] == 3.0
    assert np.isnan(out[4])


def test_sided_windows_exclude_the_frame():
    x = np.arange(6, dtype=float)
    before, after = _sided_means(x, 2)
    assert np.isnan(before[0]) and before[1] == 0 and before[3] == 1.5
    assert after[3] == 4.5 and np.isnan(after[5])
    b, a = _sided_sums(np.array([1, 0, 1, 1, 0, 1]), 2)
    assert list(b) == [0, 1, 1, 1, 2, 1] and list(a) == [1, 2, 1, 1, 1, 0]


def test_viterbi_accepts_per_frame_transitions():
    rng = np.random.default_rng(0)
    log_emit = np.log(rng.dirichlet(np.ones(3), 200))
    log_trans = np.log(np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.1, 0.8]]))
    start = np.log(np.ones(3) / 3)
    same = viterbi(log_emit, np.repeat(log_trans[None], 200, axis=0), start)
    assert np.array_equal(same, viterbi(log_emit, log_trans, start))


def _toy_match(n=400, seed=1):
    rng = np.random.default_rng(seed)
    px = rng.uniform(0, 105, (n, 22)).astype(np.float32)
    py = rng.uniform(0, 68, (n, 22)).astype(np.float32)
    px[:, :11] *= 0.5  # home stands at the low end
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int8)
    t = np.arange(n)
    return px, py, team, 50 + 10 * np.sin(t / 20.0), 34 + 5 * np.cos(t / 15.0)


def test_swapped_features_are_the_features_of_the_swapped_teams():
    px, py, team, bx, by = _toy_match()
    args = (bx, by, (105.0, 68.0), 25, [(0, 400)])
    X, names = features(px, py, team, *args, swap=True)
    Y, _ = features(px, py, 1 - team, *args)
    assert np.array_equal(np.isnan(X), np.isnan(Y))
    assert np.allclose(np.nan_to_num(X), np.nan_to_num(Y))
    gap = names.index("gap")
    Z, _ = features(px, py, team, *args)
    assert np.allclose(np.nan_to_num(Z[:, gap]), -np.nan_to_num(X[:, gap]))


def test_exported_trees_reproduce_scikit_learn(tmp_path):
    ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int) + (np.nan_to_num(X[:, 1]) > 1)
    model = ensemble.HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    path = tmp_path / "m.npz"
    trees.export(model, path)
    assert np.allclose(trees.predict_proba(trees.load(path), X), model.predict_proba(X), atol=1e-9)


def test_joint_features_leave_out_the_ball_alone_and_swap_with_the_teams():
    from hmm.control import BALL_ALONE, STRIDE, hold, joint_features
    px, py, team, bx, by = _toy_match()
    args = (bx, by, (105.0, 68.0), 25, [(0, 400)])
    X, names, frames = joint_features(px, py, team, *args)
    assert list(frames) == list(range(0, 400, STRIDE)) and X.shape == (len(frames), len(names))
    assert len(set(names)) == len(names)
    assert not [n for n in names if any(b in n for b in BALL_ALONE) or n.startswith("k_")]
    assert "gap" in names and "p_width_diff" in names and "b_gap" in names
    Xs = joint_features(px, py, team, *args, swap=True)[0]
    Y = joint_features(px, py, 1 - team, *args)[0]
    assert np.array_equal(np.isnan(Xs), np.isnan(Y))
    assert np.allclose(np.nan_to_num(Xs), np.nan_to_num(Y))
    held = hold(np.arange(len(frames)), 400)
    assert list(held[:8]) == [0, 0, 0, 1, 1, 1, 1, 1] and held[-1] == len(frames) - 1
