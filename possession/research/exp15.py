"""A turnover clue for the decode: everybody's acceleration along the pitch peaks about 0.7 s
after a turnover (exp14). Let the who chain change team more cheaply where that impulse points
the right way (an input-dependent transition cost), instead of asking the emissions to carry it."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
from concurrent.futures import ProcessPoolExecutor
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths):
    m["who"].update(feats2.cached(p)); z = np.load(os.path.join(ROOT, "data", "cache", "research4", os.path.basename(p))); m["who"].update({k: z[k] for k in z})
WHO = ["wid", "len", "vcx", "speed", "press_max", "pol"]
Xw, _ = design(ms, "who", WHO, ["s1", "s3", "d2"])
Xd, _ = design(ms, "dead", ["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])
FW = {t: fit_fold(ms, Xw, t, "who", True) for t in PROVIDERS}
FD = {t: fit_fold(ms, Xd, t, "dead", False) for t in PROVIDERS}
SD = {t: float(np.mean([np.nanstd(m["who"]["acc_x"]) for m in ms if m["prov"] != t])) for t in PROVIDERS}

def viterbi_dyn(logit, up, dn):
    """Two states (1 = home). up[t]: cost of changing to home between t-1 and t; dn[t]: to away."""
    n = len(logit); back = np.zeros((n, 2), np.int8)
    s0, s1 = -logit[0] / 2, logit[0] / 2
    lg = logit.tolist(); up = up.tolist(); dn = dn.tolist()
    for t in range(1, n):
        a = s1 - dn[t]; b = s0 - up[t]
        if a > s0: back[t, 0] = 1; n0 = a
        else: n0 = s0
        if b > s1: n1 = b
        else: back[t, 1] = 1; n1 = s1
        s0, s1 = n0 - lg[t] / 2, n1 + lg[t] / 2
    out = np.zeros(n, bool); k = int(s1 > s0)
    for t in range(n - 1, -1, -1):
        out[t] = k; k = back[t, k]
    return out

GRID = [(c, g, sh) for c in (3.0, 6.0, 10.0) for g in (0.0, 2.0, 4.0, 8.0) for sh in (0.6,)] + [(6.0, 4.0, 0.2), (6.0, 4.0, 1.0), (10.0, 8.0, 1.0), (10.0, 12.0, 0.6), (15.0, 12.0, 0.6), (15.0, 16.0, 0.6)]

def one(i):
    m = ms[i]; t = m["prov"]
    hl, dl = np.nan_to_num(apply(FW[t], Xw[i])), np.nan_to_num(apply(FD[t], Xd[i]))
    g = np.nan_to_num(per_half(lambda v: smooth(v, 0.4), m["who"]["acc_x"], m["half"])) / SD[t]
    out = []
    for c, gam, sh in GRID:
        k = int(round(sh * HZ))
        labp = np.full(len(m["labels25"]), "X", dtype="<U1")
        for h, (a, b) in enumerate(m["halves"]):
            s = m["half"] == h
            gs = np.r_[g[s][k:], np.zeros(k)]            # the impulse k samples later belongs to a change now
            up = np.maximum(c - gam * np.maximum(gs, 0), 0.5); dn = np.maximum(c - gam * np.maximum(-gs, 0), 0.5)
            home = viterbi_dyn(hl[s], up, dn)
            alive = absorb(dl[s] < 0, {True: 2 * HZ, False: 3 * HZ})
            labp[a:b + 1] = np.repeat(np.where(alive, np.where(home, "H", "A"), "D"), feats.STEP)[:b + 1 - a]
        out.append(score(m, labp))
    return np.array(out)

if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        res = np.array(list(ex.map(one, range(len(ms)))))          # (match, grid, metric)
    prov = np.array([m["prov"] for m in ms])
    pm_ = np.array([res[prov == p].mean(0) for p in PROVIDERS])       # (provider, grid, metric)
    for gi, g in enumerate(GRID):
        print(f"cost {g[0]:4.1f} clue {g[1]:4.1f} shift {g[2]:.1f}: spellF1 {pm_[:, gi, 0].mean():.3f} ({' '.join(f'{x:.3f}' for x in pm_[:, gi, 0])}) team {pm_[:, gi, 3].mean():.3f}")
    outs = []
    for pi, p in enumerate(PROVIDERS):
        others = [j for j in range(3) if j != pi]
        pick = int(pm_[others][:, :, 0].mean(0).argmax()); outs.append(pm_[pi, pick]); print("held out", p, "picked", GRID[pick], f"{pm_[pi, pick, 0]:.3f}")
        nc = [gi for gi, g in enumerate(GRID) if g[1] == 0]; pk0 = nc[int(pm_[others][:, nc, 0].mean(0).argmax())]
        print("   without the clue picked", GRID[pk0], f"{pm_[pi, pk0, 0]:.3f}")
    print("held-out mean with the clue:", np.mean(outs, 0).round(3))
