"""alldata/physics: the touch detector, the smoothing window and the team symmetry of the features."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "alldata", "physics"))
import ball as BL  # noqa: E402
import common as C  # noqa: E402
import feats as FE  # noqa: E402

FPS = 25


def path_with_one_kick(n=200, at=100):
    """A ball rolling along x at 4 m/s, kicked at frame `at` to 12 m/s along y."""
    t = np.arange(n)
    x = np.where(t < at, 20 + 4 * t / FPS, 20 + 4 * at / FPS)
    y = np.where(t < at, 30.0, 30 + 12 * (t - at) / FPS)
    return x.astype(float), y.astype(float)


def test_touch_found_at_the_kick_and_given_to_the_nearer_team():
    x, y = path_with_one_kick()
    k = BL.kin(x, y, FPS)
    d_home = np.full(len(x), 9.0)
    d_away = np.full(len(x), 9.0)
    d_away[95:105] = 0.5
    t = BL.touches(k, d_home, d_away, FPS)
    assert len(t["frame"]) == 1 and abs(t["frame"][0] - 100) <= 2
    assert t["team"][0] == -1


def test_touch_with_nobody_near_has_no_team():
    x, y = path_with_one_kick()
    far = np.full(len(x), 9.0)
    t = BL.touches(BL.kin(x, y, FPS), far, far, FPS)
    assert list(t["team"]) == [0]


def test_smoothing_keeps_a_straight_path_and_the_gaps():
    ball = np.stack([np.linspace(0, 50, 300), np.linspace(10, 40, 300), np.zeros(300)], 1)
    ball[120:150] = np.nan
    out = C.smooth(ball, 16.0)
    assert np.isnan(out[120:150, 0]).all() and np.isfinite(out[:120, 0]).all()
    assert np.allclose(out[30:90, :2], ball[30:90, :2], atol=1e-9)  # away from the edges a line is unchanged


def test_exchanging_the_teams_flips_the_home_relative_columns():
    rng = np.random.default_rng(0)
    n, K = 500, 8
    players = np.zeros((n, K, 3), np.float32)
    players[:, :, 0] = rng.uniform(10, 95, K) + np.cumsum(rng.normal(0, 0.05, (n, K)), 0)
    players[:, :, 1] = rng.uniform(5, 60, K) + np.cumsum(rng.normal(0, 0.05, (n, K)), 0)
    players[:, :, 2] = 1.0
    team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    players[:, team == 1, 0] += 5
    ball = np.stack([40 + np.cumsum(rng.normal(0, 0.2, n)), 30 + np.cumsum(rng.normal(0, 0.2, n)), np.zeros(n)], 1)
    shown = np.ones(n, bool)
    prior = np.tile([0.5, 0.3, 0.2], (n // C.STEP + 1, 1))
    F, names, _ = FE.half_features(players, ball, shown, team, prior, (105.0, 68.0), FPS)
    Fs, _, _ = FE.half_features(players, ball, shown, 1 - team, prior[:, [1, 0, 2]], (105.0, 68.0), FPS)
    for col in ("d_sd", "b_x", "t_last1_team", "p_width_diff"):
        i = names.index(col)
        assert np.allclose(np.nan_to_num(F[:, i]), -np.nan_to_num(Fs[:, i]), atol=1e-4), col
    assert np.allclose(F[:, names.index("q_home")], Fs[:, names.index("q_away")])
    assert np.allclose(np.nan_to_num(F[:, names.index("d_min")]), np.nan_to_num(Fs[:, names.index("d_min")]), atol=1e-5)
