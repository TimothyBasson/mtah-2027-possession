"""Per-frame features for a learned "who controls the ball" classifier.

The hand-built clues (hmm/clues.py, hmm/ball.py) each look at one thing: the nearest-player gap,
the ball's place, the players' speed, a touch's team. This module lays out, for every frame, the
richer picture a classifier can combine: both teams' nearest and second-nearest players, how many
of each are close, whether the nearest player moves with the ball (a carry) or not, the ball's
speed and its velocity change, the touches either side of the frame and whose they were, and
windowed averages of all that over 0.5 / 1.5 / 4 s either side (the decoding is offline).

Everything is computed from positions only, with the same code on the external matches and on
BAR-ATH, so a model trained on the former sees the same inputs on the latter.

Home / away symmetry is not built into the columns. Instead features() is called a second time with
the teams swapped (team 1 - team, and the ball's x seen from the other end), and the classifier is
trained on both copies with the labels swapped to match. At prediction time the two copies are
averaged (control_posteriors), so the output cannot favour a team.
"""

import warnings

import numpy as np

from hmm import trees
from hmm.ball import ball_velocity, detect_touches
from hmm.clues import ball_is_usable, median_player_speed, metres_outside, team_width_difference

# v8: the classifier is trained and evaluated on every STRIDE-th frame, and each score is held for
# the frames around it. Neighbouring frames repeat each other.
STRIDE = 5
# Column-name fragments joint_features() leaves out: what the ball does on its own (whether it is
# seen, how fast it moves, how long since it was seen). IDSSE freezes the ball in a stoppage and
# Metrica hides it, so a classifier given these columns learns "still or missing means dead", which
# is how those two providers record a stoppage and not football. Held out, SkillCorner pays for it
# (external/README.md).
BALL_ALONE = ("seen", "ball_speed", "bspeed", "ball_dv", "line_at_last_seen")
# The team-shape feature groups taken from seq/features.py: players only, and the ball against the
# players. Its k_ group (the ball alone) is left out for the same reason.
SHAPE_GROUPS = ("p_", "b_")
# Seconds either side of the frame for the windowed averages.
WINDOWS = (0.5, 1.5, 4.0)
# Seconds before / after the frame for the one-sided windows.
SIDED = (0.5, 1.5)
# Seconds at which the time since / until a touch is capped.
TOUCH_HORIZON = 5.0
# Metres for the "players close to the ball" counts.
NEAR = (2.0, 5.0)
# Player velocity: frames either side for the centred difference.
PLAYER_LAG = 2
# Cap on distances, metres, so a missing team does not produce inf.
FAR = 30.0
CHUNK = 20000


def _windowed_mean(x, half):
    """NaN-aware centred moving mean over 2 * half + 1 frames; NaN where nothing was seen."""
    x = np.asarray(x, float)
    seen = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(seen, x, 0.0))])
    n = np.concatenate([[0], np.cumsum(seen)])
    idx = np.arange(len(x))
    lo = np.clip(idx - half, 0, len(x))
    hi = np.clip(idx + half + 1, 0, len(x))
    count = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (c[hi] - c[lo]) / np.maximum(count, 1), np.nan)


def _sided_means(x, half):
    """(before, after): NaN-aware means over the half frames strictly before / after each frame."""
    x = np.asarray(x, float)
    seen = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(seen, x, 0.0))])
    n = np.concatenate([[0], np.cumsum(seen)])
    idx = np.arange(len(x))
    out = []
    for lo, hi in ((np.clip(idx - half, 0, None), idx),
                   (idx + 1, np.clip(idx + 1 + half, None, len(x)))):
        hi = np.maximum(hi, lo)
        count = n[hi] - n[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(count > 0, (c[hi] - c[lo]) / np.maximum(count, 1), np.nan))
    return out


def _sided_sums(x, half):
    """(before, after): sums over the half frames strictly before / after each frame."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, float))])
    idx = np.arange(len(x))
    lo_b = np.clip(idx - half, 0, None)
    hi_a = np.clip(idx + 1 + half, None, len(x))
    return c[idx] - c[lo_b], c[hi_a] - c[np.minimum(idx + 1, len(x))]


def _player_velocity(px, py, fps):
    vx = np.full(px.shape, np.nan, dtype=np.float32)
    vy = np.full(py.shape, np.nan, dtype=np.float32)
    k = PLAYER_LAG
    vx[k:-k] = (px[2 * k:] - px[:-2 * k]) * fps / (2 * k)
    vy[k:-k] = (py[2 * k:] - py[:-2 * k]) * fps / (2 * k)
    return vx, vy


def _per_team(px, py, vx, vy, ball_x, ball_y, bvx, bvy, team):
    """For each team: nearest distance, second-nearest distance, counts within NEAR, the nearest
    player's speed and his speed relative to the ball's velocity."""
    n = len(px)
    out = {}
    for side in (0, 1):
        for key in ("d1", "d2", "n_near0", "n_near1", "p_speed", "rel_v"):
            out[key, side] = np.full(n, np.nan, dtype=np.float32)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        d = np.hypot(px[lo:hi] - ball_x[lo:hi, None], py[lo:hi] - ball_y[lo:hi, None])
        tm = team[lo:hi] if team.ndim == 2 else np.broadcast_to(team, d.shape)
        rows = np.arange(hi - lo)
        for side in (0, 1):
            ds = np.where((tm == side) & np.isfinite(d), d, np.inf)
            order = np.argpartition(ds, 1, axis=1)[:, :2]
            first = np.where(ds[rows, order[:, 0]] <= ds[rows, order[:, 1]], order[:, 0],
                             order[:, 1])
            second = np.where(first == order[:, 0], order[:, 1], order[:, 0])
            d1, d2 = ds[rows, first], ds[rows, second]
            out["d1", side][lo:hi] = np.where(np.isfinite(d1), np.minimum(d1, FAR), np.nan)
            out["d2", side][lo:hi] = np.where(np.isfinite(d2), np.minimum(d2, FAR), np.nan)
            ball_seen = np.isfinite(ball_x[lo:hi])
            for j, r in enumerate(NEAR):
                cnt = np.sum(ds <= r, axis=1).astype(np.float32)
                out["n_near%d" % j, side][lo:hi] = np.where(ball_seen, cnt, np.nan)
            pvx, pvy = vx[lo:hi][rows, first], vy[lo:hi][rows, first]
            ok = np.isfinite(d1)
            out["p_speed", side][lo:hi] = np.where(ok, np.hypot(pvx, pvy), np.nan)
            out["rel_v", side][lo:hi] = np.where(
                ok, np.hypot(pvx - bvx[lo:hi], pvy - bvy[lo:hi]), np.nan)
    return out


def _touch_timing(touches, n, fps):
    """Seconds since the last and until the next touch by each team (capped), and the team of the
    last assigned touch (+1 home, -1 away, 0 none within the horizon)."""
    cap = TOUCH_HORIZON
    idx = np.arange(n)
    out = {}
    for side in (0, 1):
        f = np.sort(touches["frame"][touches["team"] == side])
        if len(f) == 0:
            out["since", side] = np.full(n, cap, np.float32)
            out["until", side] = np.full(n, cap, np.float32)
            continue
        i = np.searchsorted(f, idx, side="right") - 1
        since = np.where(i >= 0, (idx - f[np.clip(i, 0, None)]) / fps, cap)
        j = np.searchsorted(f, idx, side="left")
        until = np.where(j < len(f), (f[np.clip(j, None, len(f) - 1)] - idx) / fps, cap)
        out["since", side] = np.minimum(since, cap).astype(np.float32)
        out["until", side] = np.minimum(until, cap).astype(np.float32)
    last = np.where(out["since", 0] < out["since", 1], 1.0,
                    np.where(out["since", 1] < out["since", 0], -1.0, 0.0))
    last[(out["since", 0] >= cap) & (out["since", 1] >= cap)] = 0.0
    out["last"] = last.astype(np.float32)
    return out


def home_goal_x(px, team, halves, length):
    """Per frame, the x of the goal the home team defends (0 or length), from where each team's
    players stand on average over each half. NaN outside the halves."""
    out = np.full(len(px), np.nan)
    for lo, hi in halves:
        tm = team[lo:hi] if team.ndim == 2 else np.broadcast_to(team, px[lo:hi].shape)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mh = np.nanmedian(np.nanmean(np.where(tm == 0, px[lo:hi], np.nan), axis=1))
            ma = np.nanmedian(np.nanmean(np.where(tm == 1, px[lo:hi], np.nan), axis=1))
        out[lo:hi] = 0.0 if mh < ma else length
    return out


def control_posteriors(model, X, Xs):
    """(frames, 3) p(H, A, D) from a model saved by external/train_control.py, averaged over the
    two team orientations so that it cannot favour a team. X and Xs are features() with swap False
    and True."""
    p = trees.predict_proba(model, X)
    q = trees.predict_proba(model, Xs)[:, [1, 0, 2]]
    return (p + q) / 2


def control_log_ratio(model, X, Xs, floor=1e-6):
    """(frames, 3) log p(state | features) - log p(state): the classifier's posterior turned into
    a scaled likelihood (divided by the class shares it was trained on) for the HMM's emissions."""
    p = control_posteriors(model, X, Xs)
    return np.log(np.clip(p, floor, 1.0)) - np.log(model["prior"])


def features(px, py, team, ball_x, ball_y, pitch, fps, halves, swap=False):
    """(frames, n_features) float32 and the column names. px, py are (frames, slots) metres, NaN
    where untracked; team is (slots,) or (frames, slots) with 0 home, 1 away; ball NaN where
    unseen. halves are (start, stop) frame slices. swap=True computes the same columns with the
    teams' roles exchanged."""
    px = np.asarray(px, np.float32)
    py = np.asarray(py, np.float32)
    team = np.asarray(team).astype(np.int8)
    if swap:
        team = (1 - team).astype(np.int8)
    length = float(pitch[0])
    usable = ball_is_usable(ball_x, ball_y, pitch)
    bx = np.where(usable, ball_x, np.nan)
    by = np.where(usable, ball_y, np.nan)
    bvx, bvy = ball_velocity(bx, by, fps)
    vx, vy = _player_velocity(px, py, fps)
    pt = _per_team(px, py, vx, vy, bx, by, bvx, bvy, team)
    d_home, d_away = pt["d1", 0].astype(float), pt["d1", 1].astype(float)
    t = detect_touches(bx, by, d_home, d_away, fps)
    tt = _touch_timing(t, len(px), fps)
    dv = t["dv_frames"]
    speed = np.hypot(bvx, bvy)
    goal = home_goal_x(px, team, halves, length)
    with np.errstate(invalid="ignore"):
        x_from_home_goal = np.abs(bx - goal)
    outside = metres_outside(ball_x, ball_y, pitch)
    with np.errstate(invalid="ignore"):
        to_line = np.minimum.reduce([bx, length - bx, by, float(pitch[1]) - by])
    gap = np.clip(d_away - d_home, -FAR, FAR)
    home_nearer = np.where(np.isfinite(gap), (gap > 0).astype(float), np.nan)
    width = team_width_difference(py, team)
    p_speed = median_player_speed(px.astype(float), py.astype(float), fps)

    cols = {
        "seen": np.isfinite(bx).astype(np.float32),
        "ball_speed": speed, "ball_dv": dv, "outside": outside, "to_line": to_line,
        "x_from_home_goal": x_from_home_goal, "gap": gap,
        "median_speed": p_speed, "width": width,
        "touch_last": tt["last"],
    }
    for side, name in ((0, "h"), (1, "a")):
        for key in ("d1", "d2", "n_near0", "n_near1", "p_speed", "rel_v"):
            cols["%s_%s" % (key, name)] = pt[key, side]
        cols["since_%s" % name] = tt["since", side]
        cols["until_%s" % name] = tt["until", side]
    # Windowed context: who is nearer, how close, ball seen, touches, speed.
    for w in WINDOWS:
        half = int(round(w * fps))
        tag = "%gs" % w
        cols["gap_" + tag] = _windowed_mean(gap, half)
        cols["home_nearer_" + tag] = _windowed_mean(home_nearer, half)
        cols["seen_" + tag] = _windowed_mean(cols["seen"], half)
        cols["d1h_" + tag] = _windowed_mean(cols["d1_h"], half)
        cols["d1a_" + tag] = _windowed_mean(cols["d1_a"], half)
        cols["relvh_" + tag] = _windowed_mean(cols["rel_v_h"], half)
        cols["relva_" + tag] = _windowed_mean(cols["rel_v_a"], half)
        cols["mspeed_" + tag] = _windowed_mean(p_speed, half)
        cols["bspeed_" + tag] = _windowed_mean(speed, half)
        cols["width_" + tag] = _windowed_mean(width, half)
    # One-sided context: the same quantities over the stretch just before and just after the
    # frame. A change of possession is a contrast between the two, which a centred window blurs.
    touch_h = np.zeros(len(px), np.float32)
    touch_a = np.zeros(len(px), np.float32)
    touch_h[t["frame"][t["team"] == 0]] = 1
    touch_a[t["frame"][t["team"] == 1]] = 1
    for w in SIDED:
        half = int(round(w * fps))
        tag = "%gs" % w
        for key, x in (("gap", gap), ("home_nearer", home_nearer), ("d1h", cols["d1_h"]),
                       ("d1a", cols["d1_a"]), ("relvh", cols["rel_v_h"]),
                       ("relva", cols["rel_v_a"]), ("seen", cols["seen"])):
            before, after = _sided_means(x, half)
            cols["%s_before_%s" % (key, tag)] = before
            cols["%s_after_%s" % (key, tag)] = after
        for key, x in (("touch_h", touch_h), ("touch_a", touch_a)):
            before, after = _sided_sums(x, half)
            cols["%s_before_%s" % (key, tag)] = before
            cols["%s_after_%s" % (key, tag)] = after
    # Frames since the ball was last seen and where (distance to a line then).
    seen = np.isfinite(bx)
    last_seen = np.maximum.accumulate(np.where(seen, np.arange(len(bx)), -1))
    since_seen = np.where(last_seen >= 0, np.arange(len(bx)) - last_seen, 10 ** 6) / fps
    cols["since_seen"] = np.minimum(since_seen, 60.0)
    cols["line_at_last_seen"] = np.where(last_seen >= 0, to_line[np.clip(last_seen, 0, None)],
                                         np.nan)
    names = list(cols)
    X = np.column_stack([np.asarray(cols[k], np.float32) for k in names])
    return X, names


def joint_features(px, py, team, ball_x, ball_y, pitch, fps, halves, swap=False, stride=STRIDE):
    """(X, names, frames): the v8 classifier's inputs on every stride-th frame.

    features() above without its BALL_ALONE columns, next to the team-shape features of
    seq/features.py (how wide, deep and far up the pitch each team stands, which way both drift,
    who is goal-side of the ball). The first set reads the players nearest the ball; the second
    reads all 22, so it still speaks where the ball is missing or noisy. Arguments as features().
    """
    from seq.features import compute  # here, not at the top: seq/ imports hmm/
    X, names = features(px, py, team, ball_x, ball_y, pitch, fps, halves, swap=swap)
    keep = [i for i, n in enumerate(names) if not any(b in n for b in BALL_ALONE)]
    team = np.asarray(team)
    players = np.stack([np.asarray(px, np.float32), np.asarray(py, np.float32)], axis=-1)
    ball = np.column_stack([np.asarray(ball_x, float), np.asarray(ball_y, float)])
    usable = ball_is_usable(ball_x, ball_y, pitch)
    ball[~usable] = np.nan
    S, shape_names, frames = compute(players, ball, 1 - team if swap else team,
                                     [(lo, hi - 1) for lo, hi in halves], pitch, fps,
                                     stride=stride)
    keep_shape = [i for i, n in enumerate(shape_names) if n.startswith(SHAPE_GROUPS)]
    return (np.hstack([X[frames][:, keep], S[:, keep_shape]]),
            [names[i] for i in keep] + [shape_names[i] for i in keep_shape], frames)


def hold(values, n_frames, stride=STRIDE):
    """Per-frame copy of values given on every stride-th frame from 0: each is held for the
    stride frames centred on it."""
    return values[np.clip((np.arange(n_frames) + stride // 2) // stride, 0, len(values) - 1)]
