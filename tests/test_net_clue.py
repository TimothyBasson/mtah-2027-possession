import numpy as np

from hmm.net_clue import NET_PRIOR, log_ratio, upsample


def test_upsample_holds_the_samples_and_fills_between():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = upsample(p, 8)
    assert np.allclose(out[0], p[0]) and np.allclose(out[5], p[1])
    assert np.allclose(out[2], [0.6, 0.4, 0.0])
    assert np.allclose(out[7], p[1])  # past the last sample: held


def test_log_ratio_is_zero_at_the_prior_and_outside_the_halves():
    probs = [np.tile(NET_PRIOR, (4, 1))]
    out = log_ratio(probs, [(10, 25)], 40)
    assert out.shape == (40, 3)
    assert np.allclose(out, 0)
    sure = [np.tile([0.98, 0.01, 0.01], (4, 1))]
    out = log_ratio(sure, [(10, 25)], 40)
    assert np.all(out[:10] == 0) and np.all(out[26:] == 0)
    assert np.all(out[10:26, 0] > 1) and np.all(out[10:26, 1:] < -3)


def test_log_ratio_averages_several_nets():
    a = [np.tile([0.8, 0.1, 0.1], (3, 1))]
    b = [np.tile([0.2, 0.7, 0.1], (3, 1))]
    both = log_ratio([a, b], [(0, 10)], 11)
    mean = log_ratio([np.tile([0.5, 0.4, 0.1], (3, 1))], [(0, 10)], 11)
    assert np.allclose(both, mean)
