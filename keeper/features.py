"""Keeper-end features: one row per (frame, goal end), every STRIDE frames, for the detector.

Ball-light on purpose: BAR-ATH's tracker loses the ball in the keeper's hands, so the detector
reads mostly how the players move near a goal (keeper on the move or still, attackers backing
off, defenders spreading upfield) plus where the ball was last seen.

A row is in the gate when the ball, seen now or last seen within GATE_SECONDS, is within
GATE_METRES of that goal. Rows are labelled (when labels are given) with the state relative to
the end: 0 the team defending that goal has the ball, 1 the other team, 2 dead, -1 ungraded.
"""
import warnings

import numpy as np

STRIDE = 5
GATE_METRES = 35.0
GATE_SECONDS = 30.0
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
NEAR = 35.0


def _vel(px, py, fps):
    """Centred 1 s velocity per slot, m/s; NaN where either end is untracked or implausible."""
    lag = fps // 2
    vx = np.full(px.shape, np.nan, np.float32)
    vy = np.full(px.shape, np.nan, np.float32)
    vx[lag:-lag] = (px[2 * lag:] - px[:-2 * lag]) / (2 * lag / fps)
    vy[lag:-lag] = (py[2 * lag:] - py[:-2 * lag]) / (2 * lag / fps)
    with np.errstate(invalid="ignore"):
        bad = np.hypot(vx, vy) > 12
    vx[bad] = np.nan
    vy[bad] = np.nan
    return vx, vy


def _last_seen(bx, by, usable):
    idx = np.arange(len(bx))
    last = np.maximum.accumulate(np.where(usable, idx, -1))
    ok = last >= 0
    lx = np.where(ok, bx[np.maximum(last, 0)], np.nan)
    ly = np.where(ok, by[np.maximum(last, 0)], np.nan)
    age = np.where(ok, idx - last, 10 ** 6)
    return lx, ly, age, last


def _window(a, frames, lo, hi, fps):
    """Mean of a (per frame) over [t+lo s, t+hi s) for each t in frames, NaN-aware."""
    v = np.nan_to_num(a, nan=0.0)
    n = np.isfinite(a).astype(float)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0.0], np.cumsum(n)))
    a0 = np.clip(frames + int(lo * fps), 0, len(a))
    a1 = np.clip(frames + int(hi * fps), 0, len(a))
    s, c = cv[a1] - cv[a0], cn[a1] - cn[a0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def end_features(px, py, team, bx, by, pitch, fps, halves, labels=None):
    """team is (slots,) or (frames, slots) of 0 / 1. Returns X, names, frames, ends, gate, y."""
    length, width = pitch
    px = np.asarray(px, np.float32)
    py = np.asarray(py, np.float32)
    team = np.broadcast_to(np.asarray(team), px.shape)
    T = len(px)
    vx, vy = _vel(px, py, fps)
    spd = np.hypot(vx, vy)
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(bx) & np.isfinite(by) & (bx > -5) & (bx < length + 5) & (by > -5) & (by < width + 5)
    lx, ly, age, last = _last_seen(bx, by, usable)
    bvx, bvy = _vel(np.where(usable, bx, np.nan)[:, None], np.where(usable, by, np.nan)[:, None], fps)
    bspd = np.hypot(bvx, bvy)[:, 0]
    # which team defends the x = 0 goal, per half: lower mean x of its players
    low_team = np.zeros(T, np.int8)
    for lo, hi in halves:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m0 = np.nanmedian(np.nanmean(np.where(team[lo:hi] == 0, px[lo:hi], np.nan), axis=1))
            m1 = np.nanmedian(np.nanmean(np.where(team[lo:hi] == 1, px[lo:hi], np.nan), axis=1))
        low_team[lo:hi] = 0 if m0 < m1 else 1
    graded = np.zeros(T, bool)
    for lo, hi in halves:
        graded[lo:hi] = True
    frames = np.flatnonzero(graded)
    frames = frames[frames % STRIDE == 0]
    rows, names = [], None
    for end in (0, 1):
        gx = 0.0 if end == 0 else length
        sign = 1.0 if end == 0 else -1.0  # +x is away from this goal
        dteam = low_team if end == 0 else 1 - low_team
        dmask = team == dteam[:, None]
        amask = ~dmask & np.isfinite(px)
        dmask = dmask & np.isfinite(px)
        dg = np.hypot(px - gx, py - width / 2)
        # radial velocity away from this goal
        with np.errstate(invalid="ignore", divide="ignore"):
            rad = (vx * (px - gx) + vy * (py - width / 2)) / np.maximum(dg, 0.5)
        inbox = (np.abs(px - gx) < BOX_DEPTH) & (np.abs(py - width / 2) < BOX_HALF_WIDTH)
        near = dg < NEAR

        def per_frame(val, mask, fn):
            v = np.where(mask, val, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return fn(v, axis=1)

        # the defender nearest this goal: the keeper on most frames
        dgd = np.where(dmask, dg, np.inf)
        k = np.argmin(dgd, axis=1)
        kd = dgd[np.arange(T), k]
        kd[~np.isfinite(kd)] = np.nan
        kspd = spd[np.arange(T), k]
        kx, ky = px[np.arange(T), k], py[np.arange(T), k]
        kfwd = sign * vx[np.arange(T), k]
        n_att_box = np.sum(amask & inbox, axis=1).astype(float)
        n_def_box = np.sum(dmask & inbox, axis=1).astype(float)
        att_rad = per_frame(rad, amask & near, np.nanmean)
        def_rad = per_frame(rad, dmask & near, np.nanmean)
        att_cen = per_frame(dg, amask, np.nanmean)
        def_cen = per_frame(dg, dmask, np.nanmedian)
        med = per_frame(spd, np.isfinite(spd), np.nanmedian)
        att_spd = per_frame(spd, amask & near, np.nanmedian)
        n_near = np.sum((amask | dmask) & near, axis=1).astype(float)
        # ball, seen now and last seen
        bd_goal = np.where(usable, np.hypot(bx - gx, by - width / 2), np.nan)
        ld_goal = np.hypot(lx - gx, ly - width / 2)
        with np.errstate(invalid="ignore"):
            l_behind = (sign * (lx - gx) < 0).astype(float)
        lb = np.maximum(last, 0)
        l_kd = np.hypot(kx[lb] - lx, ky[lb] - ly)  # nearest-to-goal defender to the ball when last seen
        dball = np.hypot(px - lx[:, None], py - ly[:, None])
        l_def = per_frame(dball, dmask, np.nanmin)[lb]
        l_att = per_frame(dball, amask, np.nanmin)[lb]
        cur_kd = np.where(usable, np.hypot(kx - bx, ky - by), np.nan)
        f = frames
        cols = {
            "seen": usable[f].astype(float), "age_s": np.minimum(age[f], 60 * fps) / fps,
            "ball_goal": bd_goal[f], "last_goal": ld_goal[f], "last_behind": l_behind[f],
            "last_ky": np.abs(ly[f] - width / 2), "last_keeper": l_kd[f], "last_def": l_def[f],
            "last_att": l_att[f], "ball_keeper": cur_kd[f], "ball_speed": bspd[f],
            "keeper_goal": kd[f], "keeper_y": np.abs(ky[f] - width / 2),
            "n_att_box": n_att_box[f], "n_def_box": n_def_box[f], "n_near": n_near[f],
            "att_cen": att_cen[f], "def_cen": def_cen[f],
        }
        for nm, a in (("kspd", kspd), ("kfwd", kfwd), ("attrad", att_rad), ("defrad", def_rad),
                      ("med", med), ("attspd", att_spd)):
            for lo_s, hi_s in ((-3, 0), (0, 3), (3, 8)):
                cols["{}_{}_{}".format(nm, lo_s, hi_s)] = _window(a, f, lo_s, hi_s, fps)
        # ball over the next seconds: seen share, and seen near this goal
        near_goal = (usable & (np.nan_to_num(bd_goal, nan=1e9) < GATE_METRES)).astype(float)
        cols["seen_next3"] = _window(usable.astype(float), f, 0, 3, fps)
        cols["seen_prev3"] = _window(usable.astype(float), f, -3, 0, fps)
        cols["near_next5"] = _window(near_goal, f, 0, 5, fps)
        names = list(cols)
        X = np.column_stack([cols[n] for n in names]).astype(np.float32)
        ballnear = np.where(usable, bd_goal, ld_goal)
        gate = (ballnear[f] < GATE_METRES) & (age[f] <= GATE_SECONDS * fps)
        y = np.full(len(f), -1, np.int8)
        if labels is not None:
            lab = np.asarray(labels)[f]
            d_is_home = dteam[f] == 0
            y[lab == "D"] = 2
            y[(lab == "H") & d_is_home] = 0
            y[(lab == "A") & ~d_is_home] = 0
            y[(lab == "H") & ~d_is_home] = 1
            y[(lab == "A") & d_is_home] = 1
        rows.append((X, f, np.full(len(f), end, np.int8), gate, y, dteam[f]))
    X, fr, ends, gate, y, dteam = (np.concatenate([r[i] for r in rows]) for i in range(6))
    return X, names, fr, ends, gate, y, dteam


def hide_in_hands(bx, by, px, py, pitch, radius=1.5):
    """Ball hidden whenever a player is within radius of it inside a penalty box: BAR-ATH's tracker
    loses the ball in the keeper's hands (and in crowds at the goalmouth)."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        d = np.nanmin(np.hypot(px - bx[:, None], py - by[:, None]), axis=1)
        inbox = ((bx < BOX_DEPTH) | (bx > length - BOX_DEPTH)) & (np.abs(by - width / 2) < BOX_HALF_WIDTH)
        hide = inbox & (d < radius)
    bx, by = bx.copy(), by.copy()
    bx[hide] = np.nan
    by[hide] = np.nan
    return bx, by, hide
