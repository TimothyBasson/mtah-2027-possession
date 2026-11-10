"""The ball's own motion: its velocity from the tracked positions, and the touches in it.

A ball in flight or rolling only slows down slowly and smoothly. A touch -- a kick, a header, a
first touch that kills it, a deflection -- changes its velocity abruptly, in speed, in direction
or both. So one quantity finds them all: the change in the velocity vector across a frame,

    dv(t) = |v(after t) - v(before t)|

with v before and after each fitted over its own short window, ending and starting on frame t:
the two share only that frame, so a sharp change has its single peak where the ball turned.
A touch is a local peak of dv above TOUCH_DV. Its team is the team of the nearest player within
TOUCH_RADIUS of the ball at that frame; with nobody that close it is unassigned (a bounce, the
post, a tracking glitch). It reads the same nearest-home and nearest-away distances as the gap
clue, so the keeper rule (clues.team_by_frame) applies here as well.

Velocity is a least-squares slope over VELOCITY_WINDOW frames, which is the derivative of a
Savitzky-Golay fit, and is NaN unless the ball is seen on every frame of the window: nothing is
fitted across a gap in the tracking. A fitted speed above MAX_BALL_SPEED is no kick but the
tracker jumping to another object, and is dropped.

A flight is what follows a touch that sends the ball off at FLIGHT_SPEED or more -- any touch whose
speed after is measured, see FLIGHT_SPEED. It lasts until the next touch, until the ball is lost to
the tracking or slows below SLOW_SPEED (it has rolled out or been stopped without a sharp touch),
or for at most FLIGHT_SECONDS.

The flight clue. While the ball flies nobody controls it, and whoever is nearest is just whoever
stands under its path -- often the other team, which drags the gap clue to them early. On the
labelled external matches the possession stays with the kicker's team on 95-98% of flight frames
(external/flights.py), and the same holds after every touch. So on a flight frame the clue is the
kicker's team (or "unknown") and how long ago the kick was, in FLIGHT_PHASE_EDGES; the decoder's
table says how much each is worth, and a later phase can be worth less, since a missed touch on the
receiving end lets a flight run on into the receiver's possession. Off a flight, the flight clue is
not observed. The gap clue stays on throughout: setting it aside in flight with nobody near the
ball was tried, and moved the external scores by 0.001 at most.

The thresholds are physical or chosen against Metrica's event data (external/touches.py), never
on the public window. FLIGHT_SPEED was chosen on the 10 external matches; picked with each match
left out in turn, every fold still picks 0 and the held-out spellF1 is the same 0.445.

Tried on the external matches and not kept, since a real interception or a missed receiving touch
could leave the kicker's team holding a ball it had lost:
- a weaker flight clue (its log table times 0.3 / 0.5): spellF1 0.349 / 0.382, against 0.445 at 1;
- ending a flight once a player has stayed within 1-2 m of the ball for 3-10 frames: 0.415-0.432.
v4 also gets more of the frames right than v3 on flights that end with the ball lost (IDSSE 0.716
-> 0.859, Metrica 0.413 -> 0.768).

Known limits on BAR-ATH:
- It finds 7.5 touches a minute, a third of the external rate, because the ball is so often lost
  at a player's feet.
- 20% of its flight frames run to the FLIGHT_SECONDS cap, against 0-1% on the external matches.
  Those 47 flights move at a median 8 m/s with a player a median 2.3 m away: mostly a carry whose
  small touches stay under TOUCH_DV, so the carrier's team holds it, which is right for a carry.
- The table's unknown-kicker share (0.04) comes from random ~2 s dropouts, while BAR-ATH loses the
  ball around players, so that share fits it less well.
- The tables come from labels (DFL, Metrica events) that may hand possession over on a first touch
  more readily than the challenge's controlled-possession rule.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Frames per velocity fit (odd, 0.28 s at 25 fps), and the offset of the before / after fits from
# the frame being tested: VELOCITY_WINDOW // 2, so the before fit ends on it and the after fit
# starts on it.
VELOCITY_WINDOW = 7
STEP = VELOCITY_WINDOW // 2
# m/s. Hard shots reach about 35 m/s; faster is a tracking jump.
MAX_BALL_SPEED = 40.0
# m/s change in velocity that makes a touch, and metres within which the nearest player gets it.
TOUCH_DV = 4.0
TOUCH_RADIUS = 2.5
# m/s off the touch that makes a flight, m/s below which the ball is no longer flying, and the
# longest flight in seconds (a 70 m ball at 20 m/s takes under 4 s). FLIGHT_SPEED was meant for
# long balls (12 m/s), but on the external matches spellF1 rises steadily as it drops and levels
# off below 2 m/s (all 10: 0.318 at 16, 0.357 at 12, 0.403 at 8, 0.437 at 4, 0.446 at 0). So every
# touch with a measured speed starts one: the ball belongs to the last team to touch it until the
# next touch, and long balls are just where the nearest player misleads most.
FLIGHT_SPEED = 0.0
SLOW_SPEED = 3.0
FLIGHT_SECONDS = 5.0
# Why a flight ended.
END_TOUCH, END_LOST, END_SLOW, END_CAP = 0, 1, 2, 3
# Seconds since the kick that split a flight into phases, for the flight clue. Categories: kicker
# home in phase 0..N_PHASES-1, then kicker away in the same phases, then kicker unknown.
FLIGHT_PHASE_EDGES = np.array([0.5, 1.5])
N_PHASES = len(FLIGHT_PHASE_EDGES) + 1
UNKNOWN_KICKER = 2 * N_PHASES
N_FLIGHT_CATS = UNKNOWN_KICKER + 1


def ball_velocity(ball_x, ball_y, fps):
    """(vx, vy) per frame in m/s: the least-squares slope over VELOCITY_WINDOW frames centred on
    the frame. NaN where any frame of the window is unseen, at the array edges, and where the
    speed is above MAX_BALL_SPEED."""
    half = VELOCITY_WINDOW // 2
    offsets = np.arange(VELOCITY_WINDOW) - half
    weights = offsets / np.sum(offsets ** 2) * fps
    out = []
    for pos in (ball_x, ball_y):
        v = np.full(len(pos), np.nan)
        if len(pos) >= VELOCITY_WINDOW:
            v[half:len(pos) - half] = sliding_window_view(np.asarray(pos, float),
                                                          VELOCITY_WINDOW) @ weights
        out.append(v)
    vx, vy = out
    with np.errstate(invalid="ignore"):
        jump = np.hypot(vx, vy) > MAX_BALL_SPEED
    vx[jump] = vy[jump] = np.nan
    return vx, vy


def velocity_change(vx, vy):
    """dv per frame, m/s: |v(t + STEP) - v(t - STEP)|, NaN where either side is unmeasured."""
    dv = np.full(len(vx), np.nan)
    if len(vx) > 2 * STEP:
        dv[STEP:-STEP] = np.hypot(vx[2 * STEP:] - vx[:-2 * STEP], vy[2 * STEP:] - vy[:-2 * STEP])
    return dv


def touch_frames(dv, threshold=TOUCH_DV):
    """Frames where dv is at least threshold and the largest within STEP frames either side (the
    first of a tie). A peak next to an unmeasured frame still counts: its neighbours just cannot
    beat it."""
    filled = np.where(np.isfinite(dv), dv, -np.inf)
    padded = np.concatenate([np.full(STEP, -np.inf), filled, np.full(STEP, -np.inf)])
    windows = sliding_window_view(padded, 2 * STEP + 1)
    best = windows.max(axis=1)
    first = np.argmax(windows, axis=1) == STEP
    return np.flatnonzero((filled >= threshold) & (filled == best) & first)


def touch_team(frames, d_home, d_away, radius=TOUCH_RADIUS):
    """(team, distance) per touch frame: the team of the nearer of the nearest home and nearest
    away player to the ball, -1 where neither is within radius. d_home and d_away are per-frame
    metres, NaN where unseen, as from clues.nearest_by_team."""
    frames = np.asarray(frames, dtype=int)
    home = np.where(np.isfinite(d_home[frames]), d_home[frames], np.inf)
    away = np.where(np.isfinite(d_away[frames]), d_away[frames], np.inf)
    side = np.where(away < home, 1, 0).astype(np.int8)
    dist = np.minimum(home, away)
    side[dist > radius] = -1
    return side, np.where(np.isfinite(dist), dist, np.nan)


def detect_touches(ball_x, ball_y, d_home, d_away, fps, threshold=TOUCH_DV, radius=TOUCH_RADIUS):
    """Every touch in a match. Returns a dict of per-touch arrays: frame, team (0 home, 1 away,
    -1 unassigned), dist (metres to the nearest player), dv, and the ball's speed before and after
    (m/s), plus the per-frame vx, vy and dv they came from."""
    ball_x, ball_y = np.asarray(ball_x, float), np.asarray(ball_y, float)
    vx, vy = ball_velocity(ball_x, ball_y, fps)
    dv = velocity_change(vx, vy)
    frames = touch_frames(dv, threshold)
    side, dist = touch_team(frames, np.asarray(d_home, float), np.asarray(d_away, float), radius)
    speed = np.hypot(vx, vy)
    before = np.clip(frames - STEP, 0, len(speed) - 1)
    after = np.clip(frames + STEP, 0, len(speed) - 1)
    return {"frame": frames, "team": side, "dist": dist, "dv": dv[frames],
            "speed_before": speed[before], "speed_after": speed[after],
            "vx": vx, "vy": vy, "dv_frames": dv}


def flights(touches, fps, min_speed=FLIGHT_SPEED):
    """The flights in a match, from detect_touches' output. Returns a dict of per-flight arrays --
    start (the touch frame), stop (exclusive), team (the kicker's, -1 unassigned) and end (END_*)
    -- and per-frame in_flight (bool) and kicker (the flight's team, -1 elsewhere)."""
    speed = np.hypot(touches["vx"], touches["vy"])
    n = len(speed)
    frames, teams = touches["frame"], touches["team"]
    cap = int(round(FLIGHT_SECONDS * fps))
    out = {k: [] for k in ("start", "stop", "team", "end")}
    in_flight = np.zeros(n, dtype=bool)
    kicker = np.full(n, -1, dtype=np.int8)
    for i, (start, side, off) in enumerate(zip(frames, teams, touches["speed_after"])):
        if not off >= min_speed:
            continue
        stop, end = min(start + cap, n), END_CAP
        if i + 1 < len(frames) and frames[i + 1] < stop:
            stop, end = frames[i + 1], END_TOUCH
        after = speed[start + STEP:stop]  # from the frame the after-fit is centred on
        lost = np.flatnonzero(~np.isfinite(after))
        with np.errstate(invalid="ignore"):
            slow = np.flatnonzero(after < SLOW_SPEED)
        ends = [(hits[0], why) for hits, why in ((lost, END_LOST), (slow, END_SLOW)) if hits.size]
        if ends:
            idx, end = min(ends)
            stop = start + STEP + idx
        for key, value in (("start", start), ("stop", stop), ("team", side), ("end", end)):
            out[key].append(value)
        in_flight[start + 1:stop] = True
        kicker[start + 1:stop] = side
    result = {k: np.array(v, dtype=int) for k, v in out.items()}
    result["in_flight"], result["kicker"] = in_flight, kicker
    return result


def flight_category(f, fps):
    """Flight-clue category per frame from flights() output: kicker home in phase k is k, away is
    N_PHASES + k, unknown kicker is UNKNOWN_KICKER, and -1 off a flight."""
    n = len(f["in_flight"])
    cats = np.full(n, -1, dtype=np.int8)
    for start, stop, side in zip(f["start"], f["stop"], f["team"]):
        frames = np.arange(start + 1, stop)
        if side < 0:
            cats[frames] = UNKNOWN_KICKER
        else:
            phase = np.digitize((frames - start) / fps, FLIGHT_PHASE_EDGES)
            cats[frames] = side * N_PHASES + phase
    return cats


def mirror_flight(cats):
    """The same categories with home and away swapped, for the away team's table."""
    order = np.r_[np.arange(N_PHASES, 2 * N_PHASES), np.arange(N_PHASES), UNKNOWN_KICKER]
    return np.asarray(cats)[..., order]


def flight_clue(ball_x, ball_y, d_home, d_away, fps):
    """(flight categories, touches): touches, flights and their categories for a match."""
    t = detect_touches(ball_x, ball_y, d_home, d_away, fps)
    return flight_category(flights(t, fps), fps), t
