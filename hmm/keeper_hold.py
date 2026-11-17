"""Hand a stoppage back to the keeper's team where the decoder has read a keeper holding the ball
as dead ball (keeper_holds; hmm.decode --keeper-hold).

Some tracking loses the ball in the keeper's hands. The players slow down while he holds it, so
the decoder can call a stoppage for the whole hold. The rule looks only at the decoded stoppage
and the tracking around it:

- start: the ball, last seen at most BEFORE s before the stoppage starts, is inside the pitch and
  inside a penalty area, and it is within KEEP_NEAR m (where it was, or on its last heading
  within PROJECT s) of a player of the defending side, the team of the player nearest the goal's
  centre most often over the last DEFEND_WINDOW s;
- end: the ball, first seen again from the stoppage until AFTER s after it, has a player of that
  side as its nearest player, within END_NEAR m, inside the penalty area (BOX_MARGIN m slack) and
  more than SIX_YARD m from the goal line, so a goal kick (placed in the goal area) or a corner
  does not pass; if the ball is not seen again at all, the stoppage passes only with
  blind_ok (off by default);
- the decoder already gives the keeper's team the play after the stoppage;
- no placed ball: the ball is never seen still (under STILL m/s) for MIN_STILL s away from the
  keeper during the stoppage;
- the stoppage is at most MAX_LEN s long (the Laws allow a keeper 6 s) and has live play on
  both sides.

A stoppage that passes is labelled as the keeper's team. Nothing else changes.
"""

import numpy as np

PEN_DEPTH = 16.5
PEN_HALF_WIDTH = 20.16
SIX_YARD = 5.5

BEFORE = 6.0
PROJECT = 1.0
KEEP_NEAR = 2.0
AFTER = 3.0
END_NEAR = 3.0
BOX_MARGIN = 2.0
STILL = 1.0
MIN_STILL = 0.5
MAX_LEN = 12.0
DEFEND_WINDOW = 3.0


def ball_speed(ball_x, ball_y, fps, smooth=0.4):
    """Same as hmm.decode.ball_speed (kept here so the module has no import cycle)."""
    v = np.hypot(np.gradient(np.asarray(ball_x, float)), np.gradient(np.asarray(ball_y, float))) * fps
    w = max(1, int(round(smooth * fps)))
    ok = np.isfinite(v)
    num = np.convolve(np.where(ok, v, 0), np.ones(w), "same")
    den = np.convolve(ok.astype(float), np.ones(w), "same")
    return np.where(den > w / 2, num / np.maximum(den, 1), np.nan)


def nearest_to(xs, ys, x, y, mask=None):
    """Index of the tracked player nearest (x, y), optionally among mask; None if nobody."""
    d = np.hypot(xs - x, ys - y)
    ok = np.isfinite(d) if mask is None else np.isfinite(d) & mask
    if not ok.any():
        return None
    return int(np.argmin(np.where(ok, d, np.inf)))


def in_box(x, y, goal_x, length, width, margin=0.0):
    depth = abs(x - goal_x)
    return (depth <= PEN_DEPTH + margin and abs(y - width / 2) <= PEN_HALF_WIDTH + margin
            and -0.5 <= x <= length + 0.5)


def keeper_holds(labels, px, py, team, ball_x, ball_y, pitch, fps, before=BEFORE,
                 project=PROJECT, near=KEEP_NEAR, after=AFTER, end_near=END_NEAR,
                 still=STILL, min_still=MIN_STILL, max_len=MAX_LEN, blind_ok=False,
                 need_end=True):
    """(labels, fired): a copy of labels with each keeper-holding stoppage (see the module
    docstring) given to the keeper's team, and a list of (start, start, stop, letter) for each.
    px, py are (frames, slots) in metres, team (slots,) or (frames, slots) with 0 the home side
    (H) and 1 the away side (A)."""
    out = np.array(labels, copy=True)
    length, width = float(pitch[0]), float(pitch[1])
    n = len(out)
    team = np.asarray(team)
    if team.ndim == 1:
        team = np.broadcast_to(team, px.shape)
    ball_x = np.asarray(ball_x, float)
    ball_y = np.asarray(ball_y, float)
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)
    speed = ball_speed(ball_x, ball_y, fps)
    fired = []
    isd = out == "D"
    edges = np.flatnonzero(np.diff(np.r_[0, isd.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if s == 0 or e >= n or out[s - 1] not in ("H", "A") or out[e] not in ("H", "A"):
            continue
        if e - s > max_len * fps:
            continue
        lo = max(0, s - int(before * fps))
        pre = np.flatnonzero(seen[lo:s + 1])
        if not len(pre):
            continue
        f = lo + pre[-1]
        x, y = ball_x[f], ball_y[f]
        goal_x = 0.0 if x < length / 2 else length
        if not in_box(x, y, goal_x, length, width):
            continue
        # the defending side: the team of the player nearest the goal's centre, most often over
        # the last DEFEND_WINDOW s up to f
        votes = [0, 0]
        for j in range(max(0, f - int(DEFEND_WINDOW * fps)), f + 1, max(1, fps // 5)):
            kj = nearest_to(px[j], py[j], goal_x, width / 2)
            if kj is not None and team[j, kj] in (0, 1):
                votes[int(team[j, kj])] += 1
        if not sum(votes):
            continue
        kteam = int(np.argmax(votes))
        # the ball's last heading, from the seen frames just before f
        g0 = max(lo, f - int(0.2 * fps))
        vx = vy = 0.0
        if f > g0 and seen[g0]:
            vx, vy = (ball_x[f] - ball_x[g0]) * fps / (f - g0), (ball_y[f] - ball_y[g0]) * fps / (f - g0)
        tau = np.linspace(0, project, 11)
        bxs, bys = x + vx * tau, y + vy * tau
        keep = team[f] == kteam
        d = np.hypot(bxs[:, None] - px[f][None, :], bys[:, None] - py[f][None, :])
        d = np.where(keep[None, :] & np.isfinite(d), d, np.inf)
        if not np.min(d) <= near:
            continue
        mine = team == kteam
        # the end: where the ball is seen again
        hi = min(n, e + int(after * fps))
        a = max(s, f + 1)
        post = np.flatnonzero(seen[a:hi])
        if len(post):
            g = a + post[0]
            kg = nearest_to(px[g], py[g], ball_x[g], ball_y[g])
            ok_end = (kg is not None and team[g, kg] == kteam
                      and np.hypot(px[g, kg] - ball_x[g], py[g, kg] - ball_y[g]) <= end_near
                      and in_box(ball_x[g], ball_y[g], goal_x, length, width, BOX_MARGIN)
                      and abs(ball_x[g] - goal_x) > SIX_YARD)
            if need_end and not ok_end:
                continue
        elif not blind_ok:
            continue
        # no placed ball away from the keeper during the stoppage
        placed = 0
        for j in range(s, e):
            if seen[j] and np.isfinite(speed[j]) and speed[j] < still:
                kj = nearest_to(px[j], py[j], ball_x[j], ball_y[j], mine[j])
                far = kj is None or np.hypot(px[j, kj] - ball_x[j], py[j, kj] - ball_y[j]) > end_near
                placed = placed + 1 if far else 0
                if placed >= min_still * fps:
                    break
            else:
                placed = 0
        if placed >= min_still * fps:
            continue
        letter = "H" if kteam == 0 else "A"
        if out[e] != letter:  # after a real hold, the keeper's team plays on
            continue
        out[s:e] = letter
        fired.append((s, s, e, letter))
    return out, fired
