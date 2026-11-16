"""deep/trackers.py and external/build_soccertrack.py: the pieces that can be checked without data."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "deep"), os.path.join(ROOT, "external")]
import trackers as TR  # noqa: E402
from build_soccertrack import hold_neutral, smooth  # noqa: E402


def walk(n=500, K=6, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.05, (n, K, 2)), 0).astype(np.float32)


def test_hold_neutral():
    lab = np.array(list("NNHHNNAANDNNHNN"))
    assert "".join(hold_neutral(lab)) == "HHHHHHAAADHHHHH"
    assert "".join(hold_neutral(np.array(list("DNND")))) == "DXXD"


def test_resample_keeps_every_fifth_frame():
    xy = walk()
    for rate in (5, 10):  # which is why the model, reading every fifth frame, cannot tell these apart
        assert np.allclose(TR.resample(xy, rate)[::5], xy[::5], atol=1e-5)


def test_smooth_keeps_a_straight_run_and_missing_frames():
    xy = np.linspace(0, 10, 200, dtype=np.float32)[:, None, None].repeat(2, 1).repeat(2, 2)
    xy[50:60, 0] = np.nan
    for f in (lambda v: TR.smooth(v, 1.0), lambda v: smooth(v, 1.0)):
        out = f(xy)
        assert np.isnan(out[50:60, 0]).all() and np.isfinite(out[:50, 0]).all()
        assert np.allclose(out[20:180, 1], xy[20:180, 1], atol=1e-4)


def test_lose_takes_about_its_share_and_never_everything():
    xy = walk(20000, 22)
    out = TR.lose(xy, np.random.default_rng(1), 0.07, 5)
    gone = np.isnan(out[..., 0]).mean()
    assert 0.03 < gone < 0.12


def test_style_runs_and_keeps_shape():
    rng = np.random.default_rng(0)
    xy = walk(1600, 24)
    for _ in range(30):
        out = TR.apply(xy, TR.style(rng), rng, (0.0, 0.0))
        assert out.shape == xy.shape and np.isfinite(out).any()
