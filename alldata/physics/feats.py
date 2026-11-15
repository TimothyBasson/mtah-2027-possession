"""Per-frame evidence for possession, read from the ball first and everything else second.

For each half, at every STEP-th frame, one row of features in five groups:

  ball     where the ball is, how fast, how its velocity just changed, how near each team is to it
           now, a little earlier and a little later (centred windows: the task is offline)
  touch    the touches found in the ball's path (ball.py): the team and timing of the last three
           and the next three, how many touches in a row each of those teams strings together,
           and the touch balance over a few seconds either way. This is the organisers' control
           rule laid out as numbers: "two touches in a row by the other team" is next_run >= 2.
  gap      whether the ball is shown, how long it has been and will be missing, where it was lost
           and where it comes back (a ball lost at a touchline and back at the same spot is a
           throw-in)
  players  speeds and shape of the two teams, the stoppage signals that need no ball
  prior    the players-only ball-taught model's probabilities (deep/, bt), which carry the answer
           where the ball is missing

Everything is expressed home-relative (+ is home, - is away; x runs towards the goal home attacks),
so exchanging the teams is an exact symmetry, used to double the training rows and to average the
two readings at prediction time.
"""

import warnings

import numpy as np

import ball as BL
from common import STEP

MAX_PLAYER_SPEED = 12.0
OFFS = (-4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0, 4.0)
WINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))
NEAR = 1.5     # m: a player this close "has" the ball for the near-control share
CLEAR = 0.5    # ... if the other team's nearest is at least this much further


def _shift(x, k):
    """x[t + k] with NaN beyond the ends."""
    out = np.full(len(x), np.nan)
    if k >= 0:
        out[:len(x) - k] = x[k:] if k else x
    else:
        out[-k:] = x[:k]
    return out


class _Win:
    """NaN-aware window means over [t + lo, t + hi] frames."""

    def __init__(self, x):
        ok = np.isfinite(x)
        self.cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
        self.cn = np.r_[0, np.cumsum(ok)]
        self.n = len(x)

    def mean(self, t, lo, hi, with_share=False):
        a = np.clip(t + lo, 0, self.n)
        b = np.clip(t + hi + 1, 0, self.n)
        cnt = self.cn[b] - self.cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(cnt > 0, (self.cs[b] - self.cs[a]) / cnt, np.nan)
        if with_share:
            return m, cnt / np.maximum(b - a, 1)
        return m


def half_features(players, ball, shown, team, prior, pitch, fps, extra=None):
    """Features of one half. players (n, K, 3), ball (n, 3), shown (n,), prior (ceil(n / STEP), 3).
    Returns (F (T, nf) float32, names, touches)."""
    n = len(ball)
    L, W = pitch
    t = np.arange(0, n, STEP)
    T = len(t)
    f, names = [], []

    def add(name, col):
        names.append(name)
        f.append(np.asarray(col, np.float32))

    sec = lambda s: int(round(s * fps))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        px, py, ps = players[:, :, 0].astype(float), players[:, :, 1].astype(float), players[:, :, 2].astype(float)
        ps = np.where(ps > MAX_PLAYER_SPEED, np.nan, ps)
        H, A = team == 0, team == 1
        # home attacks +x when it defends the low-x goal: the two teams against each other
        sign = 1.0 if np.nanmean(px[:, H]) < np.nanmean(px[:, A]) else -1.0
        bx = np.where(shown, ball[:, 0], np.nan).astype(float)
        by = np.where(shown, ball[:, 1], np.nan).astype(float)
        k = BL.kin(bx, by, fps)
        dist = np.hypot(px - bx[:, None], py - by[:, None])
        dH = np.nanmin(np.where(np.isfinite(dist[:, H]), dist[:, H], np.inf), 1)
        dA = np.nanmin(np.where(np.isfinite(dist[:, A]), dist[:, A], np.inf), 1)
        dH = np.where(np.isfinite(bx), np.minimum(dH, 30.0), np.nan)
        dA = np.where(np.isfinite(bx), np.minimum(dA, 30.0), np.nan)
        touches = BL.touches(k, dH, dA, fps)

        # ---- ball --------------------------------------------------------------------------
        cx = sign * (bx - L / 2)             # + towards the goal home attacks
        cy = np.abs(by - W / 2)
        add("b_x", cx[t])
        add("b_absy", cy[t])
        add("b_side_margin", (W / 2 - cy)[t])                 # negative: over the touchline
        add("b_end_margin", (L / 2 - np.abs(cx))[t])          # negative: over a goal line
        add("b_speed", k["speed"][t])
        add("b_dv", k["dv"][t])
        add("b_acc", k["acc"][t])
        add("b_vx", (sign * k["vx"])[t])
        sd = np.clip(dA, 0, 15) - np.clip(dH, 0, 15)          # + home nearer
        dmin = np.minimum(dH, dA)
        add("d_sd", sd[t])
        add("d_min", dmin[t])
        add("d_home", dH[t])
        add("d_away", dA[t])
        near = np.where(np.isfinite(dmin), 0.0, np.nan)
        near = np.where((dH < NEAR) & (dA > dH + CLEAR), 1.0, near)
        near = np.where((dA < NEAR) & (dH > dA + CLEAR), -1.0, near)
        for o in OFFS:
            j = sec(o)
            add(f"d_sd@{o:+g}", _shift(sd, j)[t])
            add(f"d_min@{o:+g}", _shift(dmin, j)[t])
            add(f"b_speed@{o:+g}", _shift(k["speed"], j)[t])
        wsd, wnear, wsp, wdv = _Win(sd), _Win(near), _Win(k["speed"]), _Win(k["dv"])
        wabs = _Win(np.abs(near))
        for lo, hi in WINS:
            for tag, a, b in ((f"+{lo:g}_{hi:g}", sec(lo), sec(hi)), (f"-{lo:g}_{hi:g}", -sec(hi), -sec(lo))):
                add(f"w_sd{tag}", wsd.mean(t, a, b))
                add(f"w_near{tag}", wnear.mean(t, a, b))
                add(f"w_nearabs{tag}", wabs.mean(t, a, b))
                add(f"w_speed{tag}", wsp.mean(t, a, b))
        add("w_dv_c", wdv.mean(t, -sec(0.5), sec(0.5)))
        # players of each team within 3 m and 6 m of the ball
        for r in (3.0, 6.0):
            cH = np.where(np.isfinite(bx), np.sum(dist[:, H] < r, 1), np.nan)
            cA = np.where(np.isfinite(bx), np.sum(dist[:, A] < r, 1), np.nan)
            add(f"n_diff{r:g}", (cH - cA)[t])
            add(f"n_sum{r:g}", (cH + cA)[t])

        # ---- touches -----------------------------------------------------------------------
        tf, tt = touches["frame"], touches["team"]            # team +1 home, -1 away, 0 nobody near
        own = tt != 0
        of, ot = tf[own], tt[own]
        m = len(of)
        # runs of touches by one team: index of the run's first and last touch
        if m:
            brk = np.flatnonzero(np.r_[True, ot[1:] != ot[:-1]])
            rid = np.cumsum(np.r_[True, ot[1:] != ot[:-1]]) - 1
            rfirst, rlast = brk[rid], np.r_[brk[1:] - 1, m - 1][rid]
        li = np.searchsorted(of, t, "right") - 1              # last owned touch at or before t
        for j in range(3):
            i = li - j
            ok = (i >= 0) & (m > 0)
            ii = np.clip(i, 0, max(m - 1, 0))
            add(f"t_last{j + 1}_team", np.where(ok, ot[ii] if m else 0, 0))
            add(f"t_last{j + 1}_dt", np.where(ok, np.minimum((t - (of[ii] if m else 0)) / fps, 20), 20))
            i = li + 1 + j
            ok = (i < m) & (m > 0)
            ii = np.clip(i, 0, max(m - 1, 0))
            add(f"t_next{j + 1}_team", np.where(ok, ot[ii] if m else 0, 0))
            add(f"t_next{j + 1}_dt", np.where(ok, np.minimum(((of[ii] if m else 0) - t) / fps, 20), 20))
        if m:
            okl, okn = li >= 0, li + 1 < m
            il, inx = np.clip(li, 0, m - 1), np.clip(li + 1, 0, m - 1)
            # the run the last touch belongs to: touches so far, touches still to come, its span
            add("t_run_back", np.where(okl, np.minimum(il - rfirst[il] + 1, 8), 0))
            add("t_run_fwd", np.where(okl, np.minimum(rlast[il] - il, 8), 0))
            add("t_run_since", np.where(okl, np.minimum((t - of[rfirst[il]]) / fps, 60), 60))
            add("t_run_left", np.where(okl, np.minimum((of[rlast[il]] - t) / fps, 60), -1))
            # the run the next touch opens (or continues)
            add("t_nrun_len", np.where(okn, np.minimum(rlast[inx] - inx + 1, 8), 0))
            add("t_nrun_span", np.where(okn, np.minimum((of[rlast[inx]] - of[inx]) / fps, 60), 0))
            add("t_nrun_same", np.where(okl & okn, (ot[il] == ot[inx]).astype(float), np.nan))
            add("t_last_speed", np.where(okl, touches["speed_after"][own][il], np.nan))
            add("t_next_speed", np.where(okn, touches["speed_after"][own][inx], np.nan))
            add("t_last_dv", np.where(okl, touches["dv"][own][il], np.nan))
            add("t_next_dv", np.where(okn, touches["dv"][own][inx], np.nan))
            add("t_last_margin", np.where(okl, touches["margin"][own][il], np.nan))
            add("t_next_margin", np.where(okn, touches["margin"][own][inx], np.nan))
        else:
            for nm in ("t_run_back", "t_run_fwd", "t_run_since", "t_run_left", "t_nrun_len", "t_nrun_span",
                       "t_nrun_same", "t_last_speed", "t_next_speed", "t_last_dv", "t_next_dv",
                       "t_last_margin", "t_next_margin"):
                add(nm, np.full(T, np.nan))
        ev = np.zeros(n)
        np.add.at(ev, of, ot)
        evn = np.zeros(n)
        np.add.at(evn, of, 1)
        cev, cevn = np.r_[0, np.cumsum(ev)], np.r_[0, np.cumsum(evn)]
        for w in (3.0, 8.0):
            j = sec(w)
            a, b, c = np.clip(t - j, 0, n), np.clip(t + 1, 0, n), np.clip(t + j + 1, 0, n)
            add(f"t_bal-{w:g}", cev[b] - cev[a])
            add(f"t_bal+{w:g}", cev[c] - cev[b])
            add(f"t_cnt-{w:g}", cevn[b] - cevn[a])
            add(f"t_cnt+{w:g}", cevn[c] - cevn[b])

        # ---- gap ---------------------------------------------------------------------------
        idx = np.arange(n)
        last = np.maximum.accumulate(np.where(shown, idx, -1))
        nxt = np.minimum.accumulate(np.where(shown, idx, n)[::-1])[::-1]
        since = np.where(last >= 0, (idx - last) / fps, 120.0)
        until = np.where(nxt < n, (nxt - idx) / fps, 120.0)
        add("g_shown", shown[t])
        add("g_since", np.minimum(since, 120)[t])
        add("g_until", np.minimum(until, 120)[t])
        add("g_len", np.minimum(since + until, 240)[t] * (~shown[t]))
        ws = _Win(shown.astype(float))
        add("g_share2", ws.mean(t, -sec(2), sec(2)))
        add("g_share10", ws.mean(t, -sec(10), sec(10)))
        # length of the stretch of shown ball this frame sits in
        add("g_run", np.where(shown, np.minimum((nxt_missing(shown) - prev_missing(shown)) / fps, 240), 0)[t])
        lc, nc = np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)
        gone = ~shown
        for nm, src in (("x", cx), ("side_margin", W / 2 - cy), ("end_margin", L / 2 - np.abs(cx)),
                        ("speed", k["speed_edge"]), ("d_sd", sd), ("d_min", dmin)):
            add(f"g_lost_{nm}", np.where(gone & (last >= 0), src[lc], np.nan)[t])
            add(f"g_back_{nm}", np.where(gone & (nxt < n), src[nc], np.nan)[t])
        jump = np.hypot(bx[nc] - bx[lc], by[nc] - by[lc])
        add("g_jump", np.where(gone & (last >= 0) & (nxt < n), jump, np.nan)[t])
        # the ball's outward speed towards the nearer line when it was last seen
        vout = np.maximum(np.sign(by - W / 2) * k["vy_edge"], np.sign(cx) * sign * k["vx_edge"])
        add("g_lost_vout", np.where(gone & (last >= 0), vout[lc], np.nan)[t])

        # ---- players -----------------------------------------------------------------------
        nH, nA = np.sum(np.isfinite(px[:, H]), 1), np.sum(np.isfinite(px[:, A]), 1)
        msp = np.nanmean(ps, 1)
        slow = np.nanmean(np.where(np.isfinite(ps), ps < 1.0, np.nan), 1)
        fast = np.nanmean(np.where(np.isfinite(ps), ps > 4.0, np.nan), 1)
        sH, sA = np.nanmean(ps[:, H], 1), np.nanmean(ps[:, A], 1)
        mxH, mxA = sign * (np.nanmean(px[:, H], 1) - L / 2), sign * (np.nanmean(px[:, A], 1) - L / 2)
        wdH, wdA = np.nanstd(py[:, H], 1), np.nanstd(py[:, A], 1)
        lnH, lnA = np.nanstd(px[:, H], 1), np.nanstd(px[:, A], 1)
        cen = (mxH + mxA) / 2
        add("p_n_home", nH[t])
        add("p_n_away", nA[t])
        add("p_centre_x", cen[t])
        add("p_centre_gap", (mxH - mxA)[t])
        j1 = sec(1.0)
        add("p_centre_v", (_shift(cen, j1) - _shift(cen, -j1))[t] / 2)
        add("p_ball_ahead", (cx - cen)[t])
        for nm, v in (("p_speed", msp), ("p_slow", slow), ("p_fast", fast), ("p_speed_diff", sH - sA),
                      ("p_width_diff", wdH - wdA), ("p_length_diff", lnH - lnA),
                      ("p_spread", (wdH + wdA + lnH + lnA) / 4)):
            wv = _Win(v)
            add(nm, wv.mean(t, -sec(0.4), sec(0.4)))
            add(nm + "-", wv.mean(t, -sec(4), -sec(1)))
            add(nm + "+", wv.mean(t, sec(1), sec(4)))
        # the nearest players to where the ball was lost: is somebody standing over a dead ball?
        add("p_time", t / fps)
        add("p_time_left", (n - t) / fps)

        # ---- prior -------------------------------------------------------------------------
        if prior is not None:
            p = np.asarray(prior, float)[:T]
            if len(p) < T:
                p = np.r_[p, np.repeat(p[-1:], T - len(p), 0)]
            for c, nm in enumerate(("home", "away", "dead")):
                add(f"q_{nm}", p[:, c])
            ph = p[:, 0] - p[:, 1]
            for o in (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0):
                j = int(round(o * fps / STEP))
                add(f"q_ha@{o:+g}", _shift(ph, j))
                add(f"q_dead@{o:+g}", _shift(p[:, 2], j))
            wq, wd = _Win(ph), _Win(p[:, 2])
            j5 = int(round(5 * fps / STEP))
            add("q_ha-5", wq.mean(np.arange(T), -j5, 0))
            add("q_ha+5", wq.mean(np.arange(T), 0, j5))
            add("q_dead-5", wd.mean(np.arange(T), -j5, 0))
            add("q_dead+5", wd.mean(np.arange(T), 0, j5))
        if extra is not None:
            for nm, col in extra(t, sign).items():
                add(nm, col)
    return np.stack(f, 1), names, touches


def prev_missing(shown):
    idx = np.arange(len(shown))
    return np.maximum.accumulate(np.where(~shown, idx, -1))


def nxt_missing(shown):
    idx = np.arange(len(shown))
    return np.minimum.accumulate(np.where(~shown, idx, len(shown))[::-1])[::-1]


def match_features(d, shown, prior, extra=None):
    """All halves of a match: F, frames (video frame of each row), half index of each row, names."""
    Fs, frs, hs = [], [], []
    for h, (a, b) in enumerate(d["halves"]):
        ex = (lambda t, sign, h=h, a=a: extra(h, a, t, sign)) if extra is not None else None
        F, names, _ = half_features(d["players"][a:b + 1], d["ball"][a:b + 1], shown[a:b + 1], d["team"],
                                    None if prior is None else prior[h], d["pitch"], d["fps"], ex)
        Fs.append(F)
        frs.append(a + np.arange(0, b - a + 1, STEP))
        hs.append(np.full(len(F), h))
    return np.concatenate(Fs), np.concatenate(frs), np.concatenate(hs), names
