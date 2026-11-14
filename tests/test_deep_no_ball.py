"""The learned possession model may be taught by the ball but may never read it. These tests fail
if the ball of the match being labelled can reach its predictions by any path in deep/."""

import inspect
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deep"))
import data as D  # noqa: E402
from net import PossessionNet  # noqa: E402
from train import predict_half  # noqa: E402


def _match(seed=0, n=1500, k=30):
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 0.1, (n, k, 2)), 0) + rng.uniform([0, 0], [105, 68], (k, 2))
    players = np.concatenate([xy, np.abs(rng.normal(3, 1, (n, k, 1)))], -1).astype(np.float32)
    players[rng.random((n, k)) < 0.05] = np.nan
    team = (np.arange(k) % 2).astype(np.int8)
    ball = np.concatenate([np.cumsum(rng.normal(0, 0.5, (n, 2)), 0) + 50, np.ones((n, 1))], -1).astype(np.float32)
    return players, ball, team


def _other_balls(ball):
    rng = np.random.default_rng(1)
    yield np.full_like(ball, np.nan)
    yield rng.uniform(0, 100, ball.shape).astype(np.float32)
    yield ball[::-1].copy()


def test_inputs_do_not_depend_on_the_ball():
    players, ball, team = _match()
    X, M, _, fr = D.half_tensors(players, ball, team, 0, len(players) - 1, (105.0, 68.0))
    for other in _other_balls(ball):
        X2, M2, _, fr2 = D.half_tensors(players, other, team, 0, len(players) - 1, (105.0, 68.0))
        assert np.array_equal(X, X2) and np.array_equal(M, M2) and np.array_equal(fr, fr2)


@pytest.mark.parametrize("ptr", [False, True])
def test_predictions_do_not_depend_on_the_ball(ptr):
    players, ball, team = _match()
    torch.manual_seed(0)
    net = PossessionNet(nfeat=D.NFEAT, ptr=ptr)
    out = []
    for b in [ball, *_other_balls(ball)]:
        X, M, bxy, _ = D.half_tensors(players, b, team, 0, len(players) - 1, (105.0, 68.0))
        out.append(predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy)))
    assert all(np.array_equal(out[0], o) for o in out[1:])


def test_the_network_is_only_handed_the_players():
    assert list(inspect.signature(PossessionNet.forward).parameters) == ["self", "x", "mask"]
    assert D.NFEAT == 8  # x, y, vx, vy, speed, change of speed, team, has-velocity: nothing of the ball
