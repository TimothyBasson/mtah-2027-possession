"""The ball's velocity and the touches in it."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.ball import (  # noqa: E402
    END_CAP, END_LOST, END_SLOW, END_TOUCH, FLIGHT_SECONDS, MAX_BALL_SPEED, N_FLIGHT_CATS,
    N_PHASES, STEP, TOUCH_DV, UNKNOWN_KICKER, VELOCITY_WINDOW, ball_velocity, detect_touches,
    flight_category, flights, mirror_flight, touch_frames, touch_team, velocity_change)

FPS = 25


def _path(speeds, start=(10.0, 30.0), heading=(1.0, 0.0)):
    """Ball positions moving at speeds[i] m/s along heading on frame i."""
    step = np.asarray(speeds, float)[:, None] / FPS * np.asarray(heading)
    xy = np.asarray(start) + np.vstack([[0.0, 0.0], np.cumsum(step[:-1], axis=0)])
    return xy[:, 0], xy[:, 1]


def test_velocity_is_the_slope_and_nan_at_the_edges_and_across_a_gap():
    x, y = _path(np.full(40, 10.0), heading=(0.6, 0.8))
    x[20] = np.nan
    vx, vy = ball_velocity(x, y, FPS)
    half = VELOCITY_WINDOW // 2
    measured = np.isfinite(vx)
    assert not measured[:half].any() and not measured[-half:].any()
    assert not measured[20 - half:20 + half + 1].any()  # every window touching the gap
    assert np.allclose(vx[measured], 6.0) and np.allclose(vy[measured], 8.0)


def test_velocity_above_the_cap_is_a_tracking_jump_and_dropped():
    x, y = _path(np.full(20, MAX_BALL_SPEED + 5))
    vx, vy = ball_velocity(x, y, FPS)
    assert np.isnan(vx).all() and np.isnan(vy).all()


def test_a_kick_from_rest_is_one_touch_on_the_kick_frame():
    speeds = np.r_[np.zeros(30), np.full(30, 20.0)]  # still, then kicked on frame 30
    x, y = _path(speeds)
    dv = velocity_change(*ball_velocity(x, y, FPS))
    assert touch_frames(dv).tolist() == [30]
    assert np.isclose(dv[30], 20.0)


def test_a_turn_at_constant_speed_is_a_touch_and_a_straight_roll_is_not():
    x1, y1 = _path(np.full(30, 10.0))
    x2, y2 = _path(np.full(30, 10.0), start=(x1[-1] + 10.0 / FPS, y1[-1]), heading=(0.0, 1.0))
    dv = velocity_change(*ball_velocity(np.r_[x1, x2], np.r_[y1, y2], FPS))
    assert touch_frames(dv).tolist() == [30]
    assert np.isclose(dv[30], np.hypot(10.0, 10.0))
    x, y = _path(np.linspace(15.0, 5.0, 60))  # slowing down smoothly
    assert touch_frames(velocity_change(*ball_velocity(x, y, FPS))).size == 0


def test_touch_frames_needs_the_threshold_and_keeps_one_peak_per_window():
    dv = np.full(40, np.nan)
    dv[5:35] = 1.0
    dv[10], dv[11] = TOUCH_DV + 1, TOUCH_DV + 1   # a tie: the first is kept
    dv[11 + STEP + 1] = TOUCH_DV + 0.5           # far enough from the first peak to count
    dv[30] = TOUCH_DV - 0.1                       # below the threshold
    assert touch_frames(dv).tolist() == [10, 11 + STEP + 1]
    assert touch_frames(dv, threshold=TOUCH_DV - 0.5).tolist() == [10, 11 + STEP + 1, 30]


def test_touch_team_is_the_nearer_team_within_the_radius():
    d_home = np.array([1.0, 20.0, np.nan, 3.0])
    d_away = np.array([1.5, 10.0, 0.5, np.nan])
    side, dist = touch_team([0, 1, 2, 3], d_home, d_away)
    assert side.tolist() == [0, -1, 1, -1]  # frame 1: nearest 10 m away; frame 3: 3 m
    assert np.allclose(dist, [1.0, 10.0, 0.5, 3.0])


def _kicked(speeds):
    """Touches on a straight path with the nearest away player always 1 m from the ball and the
    nearest home player 30 m away."""
    x, y = _path(speeds)
    n = len(x)
    return detect_touches(x, y, np.full(n, 30.0), np.full(n, 1.0), FPS)


def test_detect_touches_finds_the_kick_and_its_team():
    t = _kicked(np.r_[np.zeros(30), np.full(30, 20.0)])
    assert t["frame"].tolist() == [30] and t["team"].tolist() == [1]
    assert np.isclose(t["speed_before"][0], 0.0) and np.isclose(t["speed_after"][0], 20.0)


def test_a_hard_kick_flies_until_the_next_touch_and_belongs_to_the_kicker():
    t = _kicked(np.r_[np.zeros(20), np.full(30, 20.0), np.zeros(20)])  # kicked, then stopped
    f = flights(t, FPS)
    # the stop at frame 50 is a touch too, whose own "flight" ends at once with the ball slow
    assert f["start"].tolist() == [20, 50] and f["stop"].tolist() == [50, 50 + STEP]
    assert f["end"].tolist() == [END_TOUCH, END_SLOW] and f["team"].tolist() == [1, 1]
    assert np.flatnonzero(f["in_flight"]).tolist() == list(range(21, 50)) + [51, 52]
    assert (f["kicker"][21:50] == 1).all() and (f["kicker"][:21] == -1).all()


def test_a_touch_slower_than_min_speed_is_no_flight():
    t = _kicked(np.r_[np.zeros(20), np.full(30, 6.0)])
    assert len(t["frame"]) == 1
    assert flights(t, FPS, min_speed=8.0)["start"].size == 0
    assert flights(t, FPS)["start"].size == 1  # FLIGHT_SPEED takes every measured touch


def test_a_flight_ends_where_the_ball_is_lost():
    speeds = np.r_[np.zeros(20), np.full(40, 20.0)]
    x, y = _path(speeds)
    x[35:], y[35:] = np.nan, np.nan
    n = len(x)
    d_away = np.full(n, 30.0)
    d_away[20] = 1.0
    t = detect_touches(x, y, np.full(n, 30.0), d_away, FPS)
    f = flights(t, FPS)
    # the last measured velocity is centred VELOCITY_WINDOW // 2 frames before the gap
    assert f["end"].tolist() == [END_LOST] and f["stop"].tolist() == [35 - VELOCITY_WINDOW // 2]


def test_a_flight_ends_when_the_ball_slows_or_at_the_cap():
    t = _kicked(np.r_[np.zeros(20), np.linspace(14.0, 1.0, 80)])  # dies away without a touch
    f = flights(t, FPS)
    assert f["end"].tolist() == [END_SLOW]
    t = _kicked(np.r_[np.zeros(20), np.full(int(FLIGHT_SECONDS * FPS) + 40, 20.0)])
    f = flights(t, FPS)
    assert f["end"].tolist() == [END_CAP]
    assert (f["stop"] - f["start"]).tolist() == [int(FLIGHT_SECONDS * FPS)]


def test_flight_category_is_the_kickers_team_by_phase_and_unknown_without_a_kicker():
    # Two flights: away kicks at frame 0 and flies 2 s; an unassigned touch at 60 flies 0.4 s.
    f = {"start": np.array([0, 60]), "stop": np.array([50, 70]), "team": np.array([1, -1]),
         "in_flight": np.zeros(80, dtype=bool)}
    cats = flight_category(f, FPS)
    assert cats[0] == -1 and (cats[50:61] == -1).all() and (cats[70:] == -1).all()
    assert cats[1] == N_PHASES + 0      # 0.04 s after the kick
    assert cats[20] == N_PHASES + 1     # 0.8 s
    assert cats[45] == N_PHASES + 2     # 1.8 s
    assert (cats[61:70] == UNKNOWN_KICKER).all()


def test_mirror_flight_swaps_the_teams_and_keeps_unknown():
    cats = np.arange(N_FLIGHT_CATS)
    assert mirror_flight(cats).tolist() == (list(range(N_PHASES, 2 * N_PHASES))
                                            + list(range(N_PHASES)) + [UNKNOWN_KICKER])
    assert mirror_flight(mirror_flight(cats)).tolist() == cats.tolist()
