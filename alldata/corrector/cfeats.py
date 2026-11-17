"""Rows for the second-stage corrector: what the main net does not see.

The main net (alldata/learned) reads, per frame, each team's nearest distances to the ball. It
never follows one player. These rows add, at the same 5 fps:

  hold   per team, the best single player's mean distance to the ball over windows before,
         around and after the frame (one player staying on the ball, not a relay of nearest ones)
  vel    how well that player's velocity matches the ball's
  touch  the three touches before and after (touch.py): how long ago, how hard, which team's
         player was beside the corrected place of the touch and by what margin, speed out
  main   the main model's probabilities (both heads) here and at lags up to 5 s, and the decoded
         run the frame sits in (its team, its length, how far the frame is from its ends)

Team columns come in pairs h_* / a_* and signed columns flip with the team swap, so `swap`
gives the rows of the same match with the teams exchanged.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "deep")]
import decode
import touch as T

import data as D

STEP = D.STEP
FAR = 30.0
LAGS = (1, 2, 4, 8, 15, 25)  # rows (0.2 s each)
NT = 3


def _wmean(cs, lo, hi, n):
    """Mean of a cumulative-summed (n + 1, K) series over frames [t + lo, t + hi]."""
    t = np.arange(n)
    a, b = np.clip(t + lo, 0, n), np.clip(t + hi + 1, 0, n)
    return (cs[b] - cs[a]) / np.maximum(b - a, 1)[:, None]


def half_rows(players, bxy, team, a, b, pitch, fps, probs, fprobs):
    """(T, F) rows and names for frames a..b. bxy (n, 2) is the ball the model is shown."""
    L, W = float(pitch[0]), float(pitch[1])
    xy = players[a:b + 1, :, :2].astype(np.float64) - np.array([L / 2, W / 2])
    ball = bxy[a:b + 1].astype(np.float64) - np.array([L / 2, W / 2])
    if not D.home_defends_left(players, team, a, b):
        xy, ball = -xy, -ball
    n = len(xy)
    idx = np.arange(n)
    seen = np.isfinite(ball[:, 0])
    okp = np.isfinite(xy[:, :, 0])
    dist = np.hypot(xy[:, :, 0] - ball[:, None, 0], xy[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), np.minimum(dist, FAR), FAR)
    pv = np.zeros_like(xy)
    k = 3
    pv[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2 * k))
    pv = np.nan_to_num(pv)
    bv, ba = T.kinematics(ball, fps)
    bvs = np.zeros_like(ball)
    bvs[k:-k] = (ball[2 * k:] - ball[:-2 * k]) * (fps / (2 * k))
    bvs = np.nan_to_num(bvs)
    rel = np.hypot(pv[:, :, 0] - bvs[:, None, 0], pv[:, :, 1] - bvs[:, None, 1])
    rel = np.where(dist < 5, np.minimum(rel, 15), 15.0)
    cs_d = np.vstack([np.zeros((1, dist.shape[1])), np.cumsum(dist, 0)])
    cs_r = np.vstack([np.zeros((1, dist.shape[1])), np.cumsum(rel, 0)])
    sel = np.arange(0, n, STEP)
    Tn = len(sel)
    cols, names = [], []

    def add(name, x):
        x = np.asarray(x, np.float32)
        cols.append(x[sel] if len(x) == n else x)
        names.append(name)

    sp = np.hypot(*bvs.T)
    add("seen", seen)
    add("speed", sp)
    add("acc", np.nan_to_num(np.hypot(*ba.T)))
    add("s_bx", np.nan_to_num(ball[:, 0]))
    add("absy", np.nan_to_num(np.abs(ball[:, 1])))
    f1, f2, f4 = int(0.4 * fps), int(1.0 * fps), int(2.5 * fps)
    wins = (("now", -2, 2), ("c1", -f1, f1), ("c2", -f2, f2), ("c4", -f4, f4), ("b2", -f2, 0), ("f2", 0, f2), ("b4", -f4, 0), ("f4", 0, f4),
            ("bb", -2 * f4, -f2), ("ff", f2, 2 * f4))
    per = {}
    for wn, lo, hi in wins:
        md, mr = _wmean(cs_d, lo, hi, n), _wmean(cs_r, lo, hi, n)
        for ti, tn in ((0, "h"), (1, "a")):
            m = team == ti
            j = md[:, m].argmin(1)
            per[(wn, tn)] = (md[:, m][idx, j], mr[:, m][idx, j], np.sort(md[:, m], 1)[:, 1] if m.sum() > 1 else np.full(n, FAR))
    for tn in "ha":
        for wn, _, _ in wins:
            add(f"{tn}_hold_{wn}", per[(wn, tn)][0])
        for wn in ("c1", "c2", "b2", "f2"):
            add(f"{tn}_vel_{wn}", per[(wn, tn)][1])
        add(f"{tn}_hold2_c2", per[("c2", tn)][2])
        m = team == (0 if tn == "h" else 1)
        add(f"{tn}_n3", (dist[:, m] < 3).sum(1))
        add(f"{tn}_cnt", okp[:, m].sum(1))
    for wn, _, _ in wins:
        add(f"s_margin_{wn}", per[(wn, "a")][0] - per[(wn, "h")][0])
    # touches
    tt = T.touches(ball, fps)
    tf = tt["frame"]
    nt = len(tf)
    if nt:
        def margin(pos):
            out = np.zeros((nt, 2))
            for j, t in enumerate(tf):
                lo, hi = max(t - 3, 0), min(t + 4, n)
                dd = np.hypot(xy[lo:hi, :, 0] - pos[j, 0], xy[lo:hi, :, 1] - pos[j, 1])
                dd = np.where(np.isfinite(dd), np.minimum(dd, FAR), FAR).min(0)
                out[j] = dd[team == 0].min() if (team == 0).any() else FAR, dd[team == 1].min() if (team == 1).any() else FAR
            return out
        mc, mo = margin(tt["pos"]), margin(tt["obs"])
        dvn = np.hypot(*tt["dv"].T)
        vo, vi = np.hypot(*tt["vout"].T), np.hypot(*tt["vin"].T)
        turn = 1 - (tt["vin"] * tt["vout"]).sum(1) / np.maximum(vo * vi, 1e-6)
        dvx = tt["dv"][:, 0]
    prev = np.searchsorted(tf, idx, side="right") - 1
    for q in range(NT):
        for side, ix in (("p", prev - q), ("n", prev + 1 + q)):
            has = (ix >= 0) & (ix < nt)
            c = np.clip(ix, 0, max(nt - 1, 0))
            if nt:
                dt = np.where(has, np.abs(tf[c] - idx) / fps, 20.0)
                has &= dt < 20
                g = lambda x, fill=0.0: np.where(has, x[c], fill)
                add(f"t{side}{q}_dt", np.minimum(dt, 20))
                add(f"t{side}{q}_dv", g(dvn))
                add(f"t{side}{q}_vout", g(vo))
                add(f"t{side}{q}_vin", g(vi))
                add(f"t{side}{q}_turn", g(turn))
                add(f"s_t{side}{q}_dvx", g(dvx))
                add(f"h_t{side}{q}_d", g(mc[:, 0], FAR)); add(f"a_t{side}{q}_d", g(mc[:, 1], FAR))
                add(f"s_t{side}{q}_m", g(np.clip(mc[:, 1] - mc[:, 0], -8, 8)))
                add(f"s_t{side}{q}_mo", g(np.clip(mo[:, 1] - mo[:, 0], -8, 8)))
            else:
                for nm, fill in (("dt", 20), ("dv", 0), ("vout", 0), ("vin", 0), ("turn", 0)):
                    add(f"t{side}{q}_{nm}", np.full(n, fill))
                add(f"s_t{side}{q}_dvx", np.zeros(n))
                add(f"h_t{side}{q}_d", np.full(n, FAR)); add(f"a_t{side}{q}_d", np.full(n, FAR))
                add(f"s_t{side}{q}_m", np.zeros(n)); add(f"s_t{side}{q}_mo", np.zeros(n))
    # the main model
    for nm, p in (("m", probs), ("mf", fprobs)):
        p = np.asarray(p, np.float32)[:Tn]
        if len(p) < Tn:
            p = np.pad(p, ((0, Tn - len(p)), (0, 0)), mode="edge")
        lp = np.log(np.clip(p, 1e-4, 1))
        for j, c in zip(range(3), ("h_{}", "a_{}", "{}_D")):
            add(c.format(nm) if j < 2 else f"{nm}_D", lp[:, j])
        for lag in LAGS:
            for sgn, sn in ((-1, "b"), (1, "f")):
                sh = np.clip(np.arange(Tn) + sgn * lag, 0, Tn - 1)
                add(f"h_{nm}_{sn}{lag}", lp[sh, 0]); add(f"a_{nm}_{sn}{lag}", lp[sh, 1])
                if lag in (2, 8, 25):
                    add(f"{nm}_D_{sn}{lag}", lp[sh, 2])
        # the decoded run each row sits in (fixed decode of score.py at 5 fps resolution)
        cost, mt, md_ = (4, 1, 1) if nm == "m" else (4, 2, 3)
        path = _decode(p, n, cost, mt, md_)
        e = np.flatnonzero(np.diff(path) != 0) + 1
        s0, s1 = np.r_[0, e], np.r_[e, n]
        rid = np.searchsorted(s1, idx, side="right")
        ln = (s1 - s0)[rid] / fps
        st = path[s0]
        pst = np.r_[2, st[:-1]][rid]
        nst = np.r_[st[1:], 2][rid]
        cur = path
        add(f"h_{nm}_run", cur == 0); add(f"a_{nm}_run", cur == 1)
        add(f"{nm}_runlen", np.minimum(ln, 60))
        add(f"{nm}_tostart", np.minimum((idx - s0[rid]) / fps, 30)); add(f"{nm}_toend", np.minimum((s1[rid] - idx) / fps, 30))
        add(f"h_{nm}_prev", pst == 0); add(f"a_{nm}_prev", pst == 1)
        add(f"h_{nm}_next", nst == 0); add(f"a_{nm}_next", nst == 1)
        pl_, nl_ = np.r_[0, (s1 - s0)[:-1]][rid] / fps, np.r_[(s1 - s0)[1:], 0][rid] / fps
        add(f"{nm}_prevlen", np.minimum(pl_, 60)); add(f"{nm}_nextlen", np.minimum(nl_, 60))
    return np.stack(cols, 1).astype(np.float32), names, a + sel


def _decode(p, n, cost, min_team, min_dead):
    import fast
    return fast.decode_path(np.log(np.clip(decode.upsample(p, n), 1e-6, 1)), cost, min_team, min_dead)


def swap(F, names):
    """Rows of the same frames with the teams exchanged."""
    pos = {nm: i for i, nm in enumerate(names)}
    perm = np.arange(len(names))
    sgn = np.ones(len(names), np.float32)
    for nm, i in pos.items():
        if nm[:2] in ("h_", "a_"):
            perm[i] = pos[("a_" if nm[0] == "h" else "h_") + nm[2:]]
        elif nm.startswith("s_"):
            sgn[i] = -1
    return F[:, perm] * sgn
