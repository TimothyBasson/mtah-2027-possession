"""Candidate segments for the corrector, and their rows.

A candidate is a stretch inside one decoded possession of team X that begins at a touch not
clearly X's (touch.py, on the ball the model is shown: a player of the other team Y is the
nearer to the ball or to the corrected place of the touch) and ends at one of the next few
touches: the stretch that would be a brief possession of Y if that touch means what it seems
to. Candidates from one start overlap; the corrector keeps the best-scored ones that do not.

    python alldata/corrector/segs.py [--base pw1e] [--version 1]   -> data/cache/alldata/corrector/segs_<base>_v<version>.npz
"""

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode
import touch as T

import data as D

OUT = os.path.join(ROOT, "data", "cache", "alldata", "corrector")
DEC = {"": (4, 1, 1), "f": (4, 2, 3), "a": (4, 1, 1)}  # head as built, head folded, the two averaged (the shipped file)
FAR = 30.0
NAMES = []
NEXT = 4  # how many following touches are tried as the end of a candidate


def learned(mod):
    spec = importlib.util.spec_from_file_location("alldata_" + mod, os.path.join(ROOT, "alldata", "learned", mod + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def base_path(p, n, cost, mt, md):
    import fast
    return fast.decode_path(np.log(np.clip(decode.upsample(p, n), 1e-6, 1)), cost, mt, md)


def half_candidates(players, bxy, team, a, b, pitch, fps, probs, pre, max_len=10.0, min_len=0.6):
    """Candidates of one half. Returns the decoded path (n,), a list of (start, end) frame
    offsets within the half, the team code each would become, and an (m, F) row array."""
    L, W = float(pitch[0]), float(pitch[1])
    xy = players[a:b + 1, :, :2].astype(np.float64) - np.array([L / 2, W / 2])
    ball = bxy[a:b + 1].astype(np.float64) - np.array([L / 2, W / 2])
    if not D.home_defends_left(players, team, a, b):
        xy, ball = -xy, -ball
    n = len(xy)
    path = base_path(probs, n, *DEC[pre])
    P = decode.upsample(probs, n)
    tt = T.touches(ball, fps)
    tf = tt["frame"]
    nt = len(tf)
    dist = np.hypot(xy[:, :, 0] - ball[:, None, 0], xy[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), np.minimum(dist, FAR), FAR)
    dteam = [dist[:, team == k] for k in (0, 1)]
    dmin = np.stack([d.min(1) if d.shape[1] else np.full(n, FAR) for d in dteam], 1)  # (n, 2)
    cs = [np.vstack([np.zeros((1, d.shape[1])), np.cumsum(d, 0)]) for d in dteam]
    sp = np.zeros(n)
    sp[2:-2] = np.hypot(*(ball[4:] - ball[:-4]).T) * fps / 4
    sp = np.nan_to_num(sp)

    def margin(pos):
        out = np.full((nt, 2), FAR)
        who = np.zeros((nt, 2), int)
        for j, t in enumerate(tf):
            lo, hi = max(t - 3, 0), min(t + 4, n)
            dd = np.hypot(xy[lo:hi, :, 0] - pos[j, 0], xy[lo:hi, :, 1] - pos[j, 1])
            dd = np.where(np.isfinite(dd), np.minimum(dd, FAR), FAR).min(0)
            for k in (0, 1):
                if (team == k).any():
                    out[j, k] = dd[team == k].min()
                    who[j, k] = np.flatnonzero(team == k)[dd[team == k].argmin()]
        return out, who
    if nt == 0:
        return path, [], [], np.zeros((0, 0), np.float32)
    mc, who = margin(tt["pos"])
    mo, _ = margin(tt["obs"])
    g = (mo[:, 1] < mo[:, 0]).astype(int)         # team beside the touch (smoothed ball)
    gc = (mc[:, 1] < mc[:, 0]).astype(int)        # and beside its corrected place
    dvn = np.hypot(*tt["dv"].T)
    vo, vi = np.hypot(*tt["vout"].T), np.hypot(*tt["vin"].T)
    e = np.flatnonzero(np.diff(path) != 0) + 1
    r0, r1 = np.r_[0, e], np.r_[e, n]
    rid = np.searchsorted(r1, tf, side="right")
    spans, to, rows = [], [], []
    clearX = (g == path[tf]) & (gc == path[tf])
    pairs = []
    for j in range(nt):
        X = path[tf[j]]
        if X == 2 or clearX[j]:
            continue
        # ends: each of the next touches inside the same decoded run, else the run's end
        ends = [k2 for k2 in range(j + 1, min(j + 1 + NEXT, nt)) if rid[k2] == rid[j]]
        for k2 in ends:
            pairs.append((j, k2 - 1, True))
        if len(ends) < NEXT:
            pairs.append((j, (ends[-1] if ends else j), False))
    for j, k, closed in pairs:
        X = path[tf[j]]
        Y = 1 - X
        s = tf[j]
        t = tf[k + 1] if closed else r1[rid[j]]
        if not closed and k > j:
            k = k  # the group runs through every touch left in the run
        dur = (t - s) / fps
        if min_len <= dur <= max_len:
            run0, run1 = r0[rid[j]], r1[rid[j]]
            grp = slice(j, k + 1)
            r = {}
            r["dur"] = dur
            r["ntouch"] = k + 1 - j
            r["closed"] = float(closed)
            r["nX_in"] = float(clearX[j + 1:k + 1].sum())
            r["first_amb"] = float(g[j] != gc[j])
            r["agree"] = float((g[grp] == Y).mean())
            r["agree_c"] = float((gc[grp] == Y).mean())
            mY = lambda m: m[grp, X] - m[grp, Y]
            r["m_first"], r["m_min"], r["m_max"], r["m_mean"] = mY(mo)[0], mY(mo).min(), mY(mo).max(), mY(mo).mean()
            r["mc_first"], r["mc_min"], r["mc_mean"] = mY(mc)[0], mY(mc).min(), mY(mc).mean()
            r["dY_first"], r["dX_first"] = mo[j, Y], mo[j, X]
            turn = 1 - (tt["vin"] * tt["vout"]).sum(1) / np.maximum(vo * vi, 1e-6)
            r["turn_first"], r["turn_last"], r["slow_first"] = turn[j], turn[k], vo[j] / max(vi[j], 0.5)
            r["dv_first"], r["dv_max"], r["vin_first"], r["vout_first"], r["vout_last"] = dvn[j], dvn[grp].max(), vi[j], vo[j], vo[k]
            r["sameY"] = float(len(set(who[grp, Y])) == 1)
            if closed:
                r["end_m"], r["end_dv"], r["end_dX"], r["end_clear"] = mo[k + 1, Y] - mo[k + 1, X], dvn[k + 1], mo[k + 1, X], float(clearX[k + 1])
                r["end_mc"], r["end_vin"], r["end_vout"] = mc[k + 1, Y] - mc[k + 1, X], vi[k + 1], vo[k + 1]
            else:
                r["end_m"], r["end_dv"], r["end_dX"], r["end_clear"], r["end_mc"], r["end_vin"], r["end_vout"] = 0.0, 0.0, FAR, 0.0, 0.0, 0.0, 0.0
            if j > 0 and rid[j - 1] == rid[j]:
                r["prev_m"], r["prev_dt"], r["prev_dv"] = mo[j - 1, Y] - mo[j - 1, X], (s - tf[j - 1]) / fps, dvn[j - 1]
            else:
                r["prev_m"], r["prev_dt"], r["prev_dv"] = 0.0, (s - run0) / fps, 0.0
            pY, pX, pD = P[s:t, Y], P[s:t, X], P[s:t, 2]
            r["pY_mean"], r["pY_max"], r["pY_min"], r["pD_mean"] = pY.mean(), pY.max(), pY.min(), pD.mean()
            r["pY_before"], r["pY_after"] = P[max(s - 2 * fps, 0):s + 1, Y].mean(), P[t:t + 2 * fps + 1, Y].mean() if t < n else 0.0
            # one player on the ball through the stretch
            for nm, kk in (("Y", Y), ("X", X)):
                md = (cs[kk][t] - cs[kk][s]) / max(t - s, 1)
                r[f"hold{nm}"] = md.min() if len(md) else FAR
                r[f"near{nm}"] = dmin[s:t, kk].mean()
                r[f"min{nm}"] = dmin[s:t, kk].min()
                r[f"n3{nm}"] = (dteam[kk][s:t] < 3).sum(1).mean()
            r["nearshare"] = float((dmin[s:t, Y] < dmin[s:t, X]).mean())
            seen = np.isfinite(ball[s:t, 0])
            r["seen"] = seen.mean()
            st = np.hypot(*np.diff(ball[s:t], axis=0).T)
            r["travel"] = np.nansum(st)
            r["speed_mean"], r["speed_max"] = sp[s:t].mean(), sp[s:t].max()
            bx = np.nanmean(ball[s:t, 0]) if seen.any() else 0.0
            r["ownx"] = bx if Y == 0 else -bx      # negative: in Y's own half
            r["absy"] = np.nanmean(np.abs(ball[s:t, 1])) if seen.any() else 0.0
            r["run_len"], r["from_start"], r["to_end"] = (run1 - run0) / fps, (s - run0) / fps, (run1 - t) / fps
            pr = path[run0 - 1] if run0 > 0 else 2
            nx = path[run1] if run1 < n else 2
            r["prevY"], r["prevD"], r["nextY"], r["nextD"] = float(pr == Y), float(pr == 2), float(nx == Y), float(nx == 2)
            NAMES[:] = list(r)
            rows.append(list(r.values()))
            spans.append((int(s), int(t)))
            to.append(Y)
    return path, spans, to, np.array(rows, np.float32).reshape(len(spans), -1)


def gain(pm, t, p, s, e, y, pad):
    """What relabelling frames s..e as y does to the possession matching around it: the change
    in matched possessions, averaged over the metric's IoU thresholds, less the possessions it
    adds times 0.35 (with F1 = 2 TP / (truth + predicted), a relabel pays when this is > 0 at
    an F1 of 0.7)."""
    lo, hi = max(s - pad, 0), min(e + pad, len(t))
    tt, pp = t[lo:hi], p[lo:hi]
    qq = pp.copy()
    qq[s - lo:e - lo] = y
    runs = pm.valid_runs(tt)
    g = pm.possession_spells(tt, runs)
    out = []
    for x in (pp, qq):
        pr = pm.possession_spells(x, runs)
        ov = pm.spell_overlaps(g, pr)
        out.append((np.mean([pm.matched_gt_flags(ov, len(g), th).sum() for th in pm.SPELL_IOUS]), len(pr)))
    return (out[1][0] - out[0][0]) - 0.35 * (out[1][1] - out[0][1])


def one(args):
    path, base, version, key = args
    name = os.path.basename(path)[:-4]
    import possession_metrics as pm
    from control import control_labels
    build = learned("build")
    d = dict(np.load(path))
    prov = name.split("_")[0]
    z = np.load(os.path.join(ROOT, "data", "cache", "alldata", "runs", base, prov, name + ".npz"))
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    truth = {"": d["labels"], "f": control_labels(d)}
    bxy = build.ball_version(d, name, version)
    out = {}
    tcs = {k: pm.label_codes(v) for k, v in truth.items()}
    code4 = np.array([int(v) for v in pm.label_codes(np.array(list("HADX")))])
    fps = int(d["fps"])
    for pre in ("", "f", "a"):
        pred = np.full(len(d["labels"]), 3, np.int8)
        S, R, TO, Y, U = [], [], [], {"": [], "f": []}, {"": [], "f": []}
        for i, (a, b) in enumerate(halves):
            probs = (z[f"{key}{i}"] + z[f"f{key}{i}"]) / 2 if pre == "a" else z[f"{pre}{key}{i}"]
            p, spans, to, rows = half_candidates(d["players"], bxy, d["team"], a, b, d["pitch"], fps, probs, pre)
            pred[a:b + 1] = p
            pc = code4[p]
            for (s, e), y in zip(spans, to):
                S.append((a + s, a + e))
                for tk in ("", "f"):  # against the labels as built and folded
                    Y[tk].append(float((tcs[tk][a + s:a + e] == code4[y]).mean()))
                    U[tk].append(gain(pm, tcs[tk][a:b + 1], pc, s, e, code4[y], 40 * fps))
            TO += to
            if len(rows):
                R.append(rows)
        out[pre + "pred"] = pred
        out[pre + "spans"] = np.array(S, int).reshape(-1, 2)
        out[pre + "to"] = np.array(TO, np.int8)
        for tk in ("", "f"):
            out[f"{pre}y{tk}"] = np.array(Y[tk], np.float32)
            out[f"{pre}u{tk}"] = np.array(U[tk], np.float32)
        out[pre + "rows"] = np.vstack(R) if R else np.zeros((0, len(NAMES)), np.float32)
    return name, out, list(NAMES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pw1e")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--key", default="s")
    a = ap.parse_args()
    paths = [p for p in D.external_paths() if "soccertrack" not in p]
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for name, out, names in ex.map(one, [(p, a.base, a.version, a.key) for p in paths]):
            for k, v in out.items():
                res[f"{name}|{k}"] = v
            print(name, len(out["to"]), len(out["fto"]), len(out["ato"]), flush=True)
    np.savez_compressed(os.path.join(OUT, f"segs_{a.base}_v{a.version}.npz"), names=np.array(names), **res)
