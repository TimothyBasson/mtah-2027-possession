"""external/broadcast.py on a small made-up match: python -m pytest external/test_broadcast.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import broadcast as bc  # noqa: E402


def match(n=2000, players=6, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 0.1, (n, players, 2)), axis=0).astype(np.float32) + 50
    ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0).astype(np.float32) + 50
    return dict(players=np.dstack([xy, np.zeros((n, players, 1), np.float32)]),
                ball=np.column_stack([ball, np.zeros(n, np.float32)]), fps=25,
                halves=np.array([[0, n // 2 - 1], [n // 2, n - 1]]))


def test_resample_keeps_exact_samples_and_is_linear_between():
    d = match()
    pxy, bxy = bc.broadcast_like(d, "m", ball_sigma=0, ball_seen=1)
    # every 5th frame of a half lands on a 10 fps sample that lands on a 25 fps frame
    assert np.allclose(bxy[0:1000:5], d["ball"][0:1000:5, :2], atol=1e-4)
    assert np.allclose(pxy[1000::5], d["players"][1000::5, :, :2], atol=1e-4)
    # frames 0, 2.5 and 5 are samples: frame 1 is 0.4 of the way from frame 0 to frame 2.5
    mid = (d["ball"][2, :2] + d["ball"][3, :2]) / 2
    assert np.allclose(bxy[1], d["ball"][0, :2] * 0.6 + mid * 0.4, atol=1e-4)
    assert bxy.shape == d["ball"][:, :2].shape and np.isfinite(bxy).all()


def test_gaps_are_not_filled():
    d = match()
    d["ball"][100:110] = np.nan
    _, bxy = bc.broadcast_like(d, "m", ball_sigma=0, ball_seen=1)
    assert np.isnan(bxy[100:110]).all()
    assert np.isfinite(bxy[:98]).all()


def test_bursts_reach_the_share_and_are_seeded():
    d = match(n=60000)
    _, a = bc.broadcast_like(d, "m", ball_seen=0.8)
    _, b = bc.broadcast_like(d, "m", ball_seen=0.8)
    assert np.array_equal(a, b, equal_nan=True)
    assert abs(np.isfinite(a[:, 0]).mean() - 0.8) < 0.05
    _, c = bc.broadcast_like(d, "other", ball_seen=0.8)
    assert not np.array_equal(a, c, equal_nan=True)


def test_only_far_players_are_dropped():
    d = match(n=20000)
    d["players"][:, 0, :2] = d["ball"][:, :2] + 1      # always next to the ball
    d["players"][:, 1, :2] = d["ball"][:, :2] + 60     # always far
    pxy, _ = bc.broadcast_like(d, "m", ball_seen=1, far_metres=25, far_share=0.3)
    assert np.isfinite(pxy[:, 0]).all()
    assert 0.15 < np.isnan(pxy[:, 1, 0]).mean() < 0.45


def test_unknown_option_is_refused():
    try:
        bc.broadcast_like(match(), "m", sigma=1)
    except ValueError:
        return
    raise AssertionError("no error")
