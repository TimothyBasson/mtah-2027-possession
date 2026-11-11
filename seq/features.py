"""Per-frame features for the learned possession labeller (seq/model.py).

Everything is measured in "home attacks towards +x" coordinates, per half, so one model serves
both halves and, with the teams swapped, both teams. Which way home attacks is read from the
players alone: a team's centre sits, on average over a half, on its own goal's side of the other
team's centre.

Three groups, named by prefix so a group can be left out and the loss measured:

    p_  players only: shape, movement and position of the two teams
    b_  the ball against the players: nearest-player gap, time to reach it, who touched it last
    k_  the ball on its own: where it is, how fast, how long since it was seen

Each base signal is given as its value at the frame and as its mean over the windows in WINDOWS
before and after it (the submission is produced offline, so the future may be read).

Every half is computed on its own: no window, carried value or timer reads across halftime, and
frames outside the halves are NaN. `team` is either one team per slot or a (frames, slots) map,
as hmm/clues.team_by_frame gives for BAR-ATH, where a keeper slot can change team by frame.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.ball import detect_touches  # noqa: E402

LAG = 5                 # frames either side for a player's velocity
MAX_SPEED = 12.0        # m/s; faster is a tracking jump
MIN_PLAYERS = 6         # a team's shape needs this many tracked players
REACH_SPEED = 6.0       # m/s a player closes on the ball at, for time-to-ball
REACH_LEAD = 0.5        # s a player keeps their current velocity first
CLIP_SECONDS = 30.0
WINDOWS = (1.0, 3.0, 8.0)   # seconds, each taken before and after the frame


def _nan(func, *args, **kw):
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(*args, **kw)


def sides(team, n_frames):
    """(is_home, is_away) as (frames, slots) masks, from a per-slot or per-frame team map."""
    team = np.asarray(team)
    if team.ndim == 1:
        team = np.broadcast_to(team, (n_frames, len(team)))
    return team == 0, team == 1


def attack_sign(px, is_home, is_away):
    """+1 if home attacks towards +x over these frames (one half), -1 if towards -x: a team's
    centre sits, on average, on its own goal's side of the other team's."""
    home = _nan(np.nanmean, np.where(is_home, px, np.nan), axis=1)
    away = _nan(np.nanmean, np.where(is_away, px, np.nan), axis=1)
    return -1.0 if _nan(np.nanmean, home - away) > 0 else 1.0


def velocities(px, py, fps):
    """(vx, vy) per player per frame from positions LAG frames either side; NaN where unmeasured
    or faster than MAX_SPEED."""
    vx = np.full(px.shape, np.nan, dtype=np.float32)
    vy = np.full(py.shape, np.nan, dtype=np.float32)
    dt = 2 * LAG / fps
    vx[LAG:-LAG] = (px[2 * LAG:] - px[:-2 * LAG]) / dt
    vy[LAG:-LAG] = (py[2 * LAG:] - py[:-2 * LAG]) / dt
    with np.errstate(invalid="ignore"):
        jump = np.hypot(vx, vy) > MAX_SPEED
    vx[jump] = vy[jump] = np.nan
    return vx, vy


def _since(flag, fps, forward=True):
    """Seconds since flag was last true (or until it is next true), clipped at CLIP_SECONDS."""
    n = len(flag)
    idx = np.arange(n)
    if forward:
        last = np.maximum.accumulate(np.where(flag, idx, -10 ** 9))
        out = (idx - last) / fps
    else:
        nxt = np.minimum.accumulate(np.where(flag, idx, 10 ** 9)[::-1])[::-1]
        out = (nxt - idx) / fps
    return np.minimum(out, CLIP_SECONDS)


def _carry(values, forward=True):
    """values with NaNs filled from the last finite one before (or the next one after)."""
    n = len(values)
    ok = np.isfinite(values)
    idx = np.arange(n)
    if forward:
        src = np.maximum.accumulate(np.where(ok, idx, -1))
    else:
        src = np.minimum.accumulate(np.where(ok, idx, n)[::-1])[::-1]
    good = (src >= 0) & (src < n)
    out = np.full(n, np.nan)
    out[good] = values[src[good]]
    return out


def _window_mean(x, w, before):
    """NaN-aware mean of x over the w frames before (t-w..t-1) or after (t+1..t+w) each frame."""
    x = np.asarray(x, dtype=np.float64)   # a float32 running sum drifts by ~0.1 over a half
    ok = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    n = len(x)
    idx = np.arange(n)
    lo, hi = (np.maximum(idx - w, 0), idx) if before else (np.minimum(idx + 1, n),
                                                           np.minimum(idx + 1 + w, n))
    count = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / count, np.nan)


def base_signals(players, ball, team, pitch, fps):
    """dict name -> per-frame signal over one half (the frames given are read as one unbroken
    stretch of play). Positive means "home" wherever a signal takes a side."""
    px, py = players[:, :, 0].astype(np.float32), players[:, :, 1].astype(np.float32)
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    length, width = float(pitch[0]), float(pitch[1])
    n = len(px)
    is_home, is_away = sides(team, n)
    sign = attack_sign(px, is_home, is_away)
    # home attacks +x: flip x about the centre line in a half where it does not
    ax = px if sign > 0 else length - px
    abx = bx if sign > 0 else length - bx
    vx, vy = velocities(px, py, fps)
    avx = vx * np.float32(sign)
    speed = np.hypot(vx, vy)
    s = {}

    def shape(sel):
        x, y = np.where(sel, ax, np.nan), np.where(sel, py, np.nan)
        count = np.isfinite(x).sum(axis=1)
        few = count < MIN_PLAYERS
        cx, cy = _nan(np.nanmean, x, axis=1), _nan(np.nanmean, y, axis=1)
        out = dict(cx=cx, cy=cy, sx=_nan(np.nanstd, x, axis=1), sy=_nan(np.nanstd, y, axis=1),
                   spread=_nan(np.nanmean, np.hypot(x - cx[:, None], y - cy[:, None]), axis=1),
                   vx=_nan(np.nanmean, np.where(sel, avx, np.nan), axis=1),
                   speed=_nan(np.nanmean, np.where(sel, speed, np.nan), axis=1),
                   count=count.astype(float))
        xs = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
        out["back"] = np.where(np.isfinite(xs[:, 1]), xs[:, 1], np.nan)   # second-lowest x
        xs = np.sort(np.where(np.isfinite(x), -x, np.inf), axis=1)
        out["front"] = np.where(np.isfinite(xs[:, 1]), -xs[:, 1], np.nan)  # second-highest x
        for k in out:
            if k != "count":
                out[k] = np.where(few, np.nan, out[k])
        return out

    home, away = shape(is_home), shape(is_away)
    # -- players only ---------------------------------------------------------------------------
    s["p_width_diff"] = home["sy"] - away["sy"]
    s["p_depth_diff"] = home["sx"] - away["sx"]
    s["p_spread_diff"] = home["spread"] - away["spread"]
    s["p_spread_sum"] = home["spread"] + away["spread"]
    s["p_centre_gap"] = home["cx"] - away["cx"]
    s["p_centre_x"] = 0.5 * (home["cx"] + away["cx"]) - length / 2
    s["p_drift"] = 0.5 * (home["vx"] + away["vx"])           # both teams run towards one goal
    s["p_drift_diff"] = home["vx"] - away["vx"]
    s["p_speed_diff"] = home["speed"] - away["speed"]
    s["p_speed_median"] = _nan(np.nanmedian, speed, axis=1)
    s["p_slow_share"] = _nan(np.nanmean, np.where(np.isfinite(speed), speed < 1.0, np.nan), axis=1)
    s["p_home_line"] = home["back"] - length / 2              # home's defensive line
    s["p_away_line"] = length / 2 - away["front"]             # away's, from its own goal
    s["p_line_diff"] = s["p_home_line"] - s["p_away_line"]
    s["p_tracked"] = home["count"] + away["count"]
    s["p_tracked_diff"] = home["count"] - away["count"]

    # -- the ball against the players -----------------------------------------------------------
    dist = _nan(np.hypot, px - bx[:, None].astype(np.float32), py - by[:, None].astype(np.float32))
    lead_x = px + np.nan_to_num(vx) * REACH_LEAD
    lead_y = py + np.nan_to_num(vy) * REACH_LEAD
    reach = _nan(np.hypot, lead_x - bx[:, None].astype(np.float32),
                 lead_y - by[:, None].astype(np.float32)) / REACH_SPEED

    def nearest(d, sel, k=0):
        part = np.sort(np.where(sel & np.isfinite(d), d, np.inf), axis=1)[:, k]
        return np.where(np.isfinite(part), part, np.nan)

    d_home, d_away = nearest(dist, is_home), nearest(dist, is_away)
    s["b_gap"] = np.clip(d_away - d_home, -20, 20)
    s["b_nearest"] = np.minimum(np.minimum(d_home, d_away), 30)
    s["b_gap2"] = np.clip(nearest(dist, is_away, 1) - nearest(dist, is_home, 1), -20, 20)
    s["b_reach_gap"] = np.clip(nearest(reach, is_away) - nearest(reach, is_home), -4, 4)
    seen = np.isfinite(bx)
    for radius in (5.0, 12.0):
        near = _nan(np.less, dist, radius)
        diff = ((near & is_home).sum(axis=1) - (near & is_away).sum(axis=1)).astype(float)
        s["b_near{:.0f}_diff".format(radius)] = np.where(seen, diff, np.nan)
    s["b_ahead_of_home"] = abx - home["cx"]      # the ball relative to each team's block
    s["b_ahead_of_away"] = abx - away["cx"]
    behind_home = _nan(np.nanmean, np.where(is_home & np.isfinite(ax),
                                            ax < abx[:, None], np.nan), axis=1)
    behind_away = _nan(np.nanmean, np.where(is_away & np.isfinite(ax),
                                            ax > abx[:, None], np.nan), axis=1)
    s["b_goalside_diff"] = np.where(seen, behind_away - behind_home, np.nan)
    s["b_gap_last"] = _carry(s["b_gap"], forward=True)       # through a dropout
    s["b_gap_next"] = _carry(s["b_gap"], forward=False)
    touches = detect_touches(bx, by, d_home, d_away, fps)
    side = np.zeros(n)
    owned = touches["team"] >= 0
    side[touches["frame"][owned]] = np.where(touches["team"][owned] == 0, 1.0, -1.0)
    is_touch = side != 0
    s["b_touch_last"] = np.nan_to_num(_carry(np.where(is_touch, side, np.nan), True))
    s["b_touch_next"] = np.nan_to_num(_carry(np.where(is_touch, side, np.nan), False))
    s["b_touch_since"] = _since(is_touch, fps, True)
    s["b_touch_until"] = _since(is_touch, fps, False)

    # -- the ball on its own --------------------------------------------------------------------
    s["k_x"] = abx - length / 2
    s["k_edge"] = np.minimum(np.minimum(bx, length - bx), np.minimum(by, width - by))
    s["k_speed"] = np.hypot(touches["vx"], touches["vy"])
    s["k_vx"] = touches["vx"] * sign
    s["k_dv"] = touches["dv_frames"]
    s["k_seen"] = seen.astype(float)
    s["k_seen_since"] = _since(seen, fps, True)
    s["k_seen_until"] = _since(seen, fps, False)
    s["k_edge_last"] = _carry(s["k_edge"], True)
    s["k_edge_next"] = _carry(s["k_edge"], False)
    return s


# Signals whose windowed means add nothing over the value (already carried or a timer).
NO_WINDOWS = {"b_gap_last", "b_gap_next", "b_touch_last", "b_touch_next", "b_touch_since",
              "b_touch_until", "k_seen_since", "k_seen_until", "k_edge_last", "k_edge_next",
              "p_tracked", "p_tracked_diff"}

def signal_columns(signals, fps, rows):
    """(columns [len(rows), d], names): each signal at these rows of its half, with its windows."""
    cols, names = [], []
    for name, x in signals.items():
        cols.append(x[rows])
        names.append(name)
        if name in NO_WINDOWS:
            continue
        for seconds in WINDOWS:
            w = int(round(seconds * fps))
            for before in (True, False):
                cols.append(_window_mean(x, w, before)[rows])
                names.append("{}@{}{:.0f}".format(name, "-" if before else "+", seconds))
    return np.stack(cols, axis=1).astype(np.float32), names


def compute(players, ball, team, halves, pitch, fps, stride=1, offset=0):
    """(features [m, d] float32, names, frames) for every stride-th frame from offset. halves are
    (first, last) frames, last included; each is computed on its own and rows outside them are
    NaN."""
    frames = np.arange(offset, len(players), stride)
    team = np.asarray(team)
    out, names = None, None
    for lo, hi in halves:
        lo, hi = int(lo), int(hi)
        signals = base_signals(players[lo:hi + 1], ball[lo:hi + 1],
                               team[lo:hi + 1] if team.ndim == 2 else team, pitch, fps)
        inside = np.flatnonzero((frames >= lo) & (frames <= hi))
        block, names = signal_columns(signals, fps, frames[inside] - lo)
        if out is None:
            out = np.full((len(frames), block.shape[1]), np.nan, dtype=np.float32)
        out[inside] = block
    return out, names, frames
