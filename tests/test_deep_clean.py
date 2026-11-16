"""deep/clean/labels.py: the rules that clean training labels with the recorded ball."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deep"))
from clean.labels import contradicted, last_touch_against, retime  # noqa: E402

FPS = 25


def lab(*runs):
    return np.concatenate([np.full(n, c, "<U1") for c, n in runs])


def touches(*ft):
    return dict(frame=np.array([f for f, _ in ft]), team=np.array([t for _, t in ft]))


def test_retime_moves_a_change_onto_the_gaining_teams_touch():
    y = lab(("H", 100), ("A", 100))
    out, moved = retime(y, touches((90, 1)), FPS)
    assert moved == 1 and (out[:90] == "H").all() and (out[90:] == "A").all()
    out, moved = retime(y, touches((110, 1)), FPS)
    assert moved == 1 and (out[:110] == "H").all() and (out[110:] == "A").all()


def test_retime_leaves_far_touches_stoppages_and_neighbours_alone():
    y = lab(("H", 100), ("A", 100))
    assert retime(y, touches((20, 1)), FPS)[1] == 0          # 3.2 s away
    assert retime(y, touches((95, 0)), FPS)[1] == 0          # the losing team's touch
    assert retime(lab(("H", 100), ("D", 100)), touches((95, 1)), FPS)[1] == 0
    y = lab(("H", 100), ("A", 10), ("H", 100))               # a touch beyond the short spell's end
    out, _ = retime(y, touches((115, 1)), FPS)
    assert (out == y).all() or ((out[100:110] == "A").any() and (out[110:] == "H").all())


def test_masks_only_name_live_frames_the_ball_contradicts():
    y = lab(("H", 100), ("D", 50))
    dh, da = np.full(150, 5.0), np.full(150, 0.5)            # away on the ball throughout
    bad = contradicted(y, dh, da, FPS)
    assert bad[:100].all() and not bad[100:].any()
    assert not contradicted(y, da, dh, FPS).any()            # home on the ball: nothing to mask
    m = last_touch_against(y, touches((10, 1), (40, 0)), FPS, hold=2.0)
    assert m[10:40].all() and not m[:10].any() and not m[40:].any()
