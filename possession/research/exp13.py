"""Fast individual movement cues around a turnover, added to the six-signal who model.
All antisymmetric in the home frame (home attacks +x): positive = movement that follows home winning the ball."""
import sys, os, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats, feats2, lab
from lab import *
from feats import STEP, FPS, MAX_SPEED
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore", RuntimeWarning)
CACHE = os.path.join(ROOT, "data", "cache", "research4")

def compute(path):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, os.path.basename(path))
    if os.path.exists(out):
        z = np.load(out); return {k: z[k] for k in z}
    d = np.load(path); players, team = d["players"], d["team"]
    parts = []
    for a, b in [tuple(h) for h in d["halves"]]:
        idx = np.arange(a, b + 1, STEP)
        def vel(lag):   # velocity from positions lag frames either side
            lo, hi = np.clip(idx - lag, a, b), np.clip(idx + lag, a, b)
            v = (players[hi, :, :2].astype(float) - players[lo, :, :2]) * FPS / np.maximum(hi - lo, 1)[:, None, None]
            v[np.hypot(v[..., 0], v[..., 1]) > MAX_SPEED] = np.nan
            return v
        v = vel(3)                                   # 0.24 s: as fast as the tracking allows
        xy = players[idx, :, :2].astype(float)
        if not (np.nanmean(xy[:, team == 0, 0]) < np.nanmean(xy[:, team == 1, 0])):
            v = -v
        vx, vy = v[..., 0], v[..., 1]; sp = np.hypot(vx, vy)
        n = len(idx); k = 3                          # 0.6 s apart at 5 Hz
        def lagdiff(x):                              # x(t+0.6) - x(t-0.6), per player, per s
            o = np.full(x.shape, np.nan); o[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2 * k / (FPS / STEP)); return o
        ax = np.clip(lagdiff(vx), -8, 8)
        f = {}
        f["acc_x"] = np.nanmean(ax, 1)
        order = np.argsort(-np.abs(np.nan_to_num(ax)), 1)[:, :4]
        f["acc_x_top4"] = np.nanmean(np.take_along_axis(ax, order, 1), 1)      # the four who react hardest
        f["acc_x_med"] = np.nanmedian(ax, 1)
        before, after = np.full(vx.shape, np.nan), np.full(vx.shape, np.nan)
        before[k:] = vx[:-k]; after[:-k] = vx[k:]
        flip_up = (before < -1) & (after > 1); flip_dn = (before > 1) & (after < -1)
        ok = np.isfinite(before) & np.isfinite(after)
        f["reversals"] = (flip_up.sum(1) - flip_dn.sum(1)) / np.maximum(ok.sum(1), 1)
        spb, spa = np.full(sp.shape, np.nan), np.full(sp.shape, np.nan)
        spb[k:] = sp[:-k]; spa[:-k] = sp[k:]
        onset = (spb < 4) & (spa > 5.5)
        f["sprint_onset"] = ((onset & (after > 0)).sum(1) - (onset & (after < 0)).sum(1)) / np.maximum(ok.sum(1), 1)
        brake = (spb > 4) & (spa < 2.5)
        f["brakes"] = ((brake & (before < 0)).sum(1) - (brake & (before > 0)).sum(1)) / np.maximum(ok.sum(1), 1)
        # heading: share turning more than 90 degrees, signed by which way they end up going
        dot = before * after + np.concatenate([np.full((k,) + vy.shape[1:], np.nan), vy[:-k]]) * np.concatenate([vy[k:], np.full((k,) + vy.shape[1:], np.nan)])
        turn = (dot < 0) & (spb > 1.5) & (spa > 1.5)
        f["turns"] = ((turn & (after > 0)).sum(1) - (turn & (after < 0)).sum(1)) / np.maximum(ok.sum(1), 1)
        # by line: do the two back lines step up or drop? (deepest four of each side, home frame)
        for nm, t, sgn in (("home_back", 0, 1), ("away_back", 1, -1)):
            xs = np.where(np.isfinite(xy[:, team == t, 0]), xy[:, team == t, 0] * sgn * (1 if True else 1), np.inf)
        parts.append(f)
    r = {k_: np.concatenate([p[k_] for p in parts]) for k_ in parts[0]}
    np.savez_compressed(out, **r); return r

if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    ms = load()
    with ProcessPoolExecutor(1) as ex:
        extra = list(ex.map(compute, paths))
    for m, p, e in zip(ms, paths, extra):
        m["who"].update(feats2.cached(p)); m["who"].update(e)
    WHO = ["wid", "len", "vcx", "speed", "press_max", "pol"]; V = ["s1", "s3", "d2"]
    FV = ["s0.6", "a2", "b2", "a5", "b5"]
    A, _ = design(ms, "who", WHO, V)
    Xd, _ = design(ms, "dead", ["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])
    def test(tag, Xw):
        rows, near, far = [], [], []
        for t in PROVIDERS:
            fw, fd = fit_fold(ms, Xw, t, "who", True), fit_fold(ms, Xd, t, "dead", False)
            sc, nr, fr = [], [], []
            for m, xw, xd in zip(ms, Xw, Xd):
                if m["prov"] != t: continue
                hl = apply(fw, xw); sc.append(score(m, decode(m, hl, apply(fd, xd))))
                lab_ = m["labels"]; live = np.isin(lab_, ["H", "A"])
                ch = np.flatnonzero(lab_[1:] != lab_[:-1]) + 1; e = np.r_[0, ch, len(lab_)]
                seg = np.searchsorted(e, np.arange(len(lab_)), "right") - 1
                dist = np.minimum(np.arange(len(lab_)) - e[seg], e[seg + 1] - 1 - np.arange(len(lab_))) / HZ
                ok = (hl >= 0) == (lab_ == "H")
                nr.append(ok[live & (dist < 1)].mean()); fr.append(ok[live & (dist >= 3)].mean())
            rows.append(np.mean(sc, 0)); near.append(np.mean(nr)); far.append(np.mean(fr))
        r = np.array(rows)
        print(f"{tag:34s} spellF1 {r[:,0].mean():.3f} ({' '.join(f'{x:.3f}' for x in r[:,0])}) team {r[:,3].mean():.3f} | team acc within 1 s of a change {np.mean(near):.3f}, beyond 3 s {np.mean(far):.3f}", flush=True)
    test("six who signals", A)
    for nm in extra[0]:
        test(f"  + {nm}", [np.c_[a, b] for a, b in zip(A, design(ms, "who", [nm], FV)[0])])
    test("  + all fast cues", [np.c_[a, b] for a, b in zip(A, design(ms, "who", list(extra[0]), FV)[0])])
    test("  + all fast cues, 8 views", [np.c_[a, b] for a, b in zip(A, design(ms, "who", list(extra[0]), ["s0.2", "s0.6", "a1", "b1", "a2", "b2", "a5", "b5"])[0])])
