"""Explicit-duration (semi-Markov) decode vs fixed minimum runs, and stoppage anticipation views."""
import sys, os, glob, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
from concurrent.futures import ProcessPoolExecutor
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
V = ["s1", "s3", "d2"]
WHO = ["wid", "len", "vcx", "speed", "press_max", "pol"]
D5 = ["corner_min", "box_max", "spread_sum", "own_half", "sp_p80"]
Xw, _ = design(ms, "who", WHO, V)
def dmat(blocks):
    return [np.column_stack(c) for c in zip(*[design(ms, "dead", n, v)[0] for n, v in blocks])]
Xd = dmat([(["sp_med", "acc_p80"], ["s0.5", "d2", "s3"]), (D5, V)])
Xd_ant = dmat([(["sp_med", "acc_p80"], ["s0.5", "d2", "s3", "a3", "b3"]), (D5, V)])
Xd_ant2 = dmat([(["sp_med", "acc_p80"], ["s0.5", "a1.5", "b1.5", "a4", "b4"]), (D5, V)])
DMAX = 90 * HZ
S = "HAD"

def durations(test):
    """log p(length) per state from the other providers' true runs, each provider equal; and the
    log transition matrix between different states."""
    hist = np.zeros((3, DMAX + 1)); T = np.zeros((3, 3))
    for p in PROVIDERS:
        if p == test: continue
        h = np.zeros((3, DMAX + 1)); t = np.zeros((3, 3))
        for m in ms:
            if m["prov"] != p: continue
            for hf in (0, 1):
                l = m["labels"][m["half"] == hf]; prev = None
                for k, g in itertools.groupby(l):
                    n = len(list(g))
                    if k in S:
                        h[S.index(k), min(n, DMAX)] += 1
                        if prev is not None and prev in S: t[S.index(prev), S.index(k)] += 1
                    prev = k
        hist += h / h.sum(1, keepdims=True); T += t / t.sum(1, keepdims=True)
    k = np.ones(5) / 5
    hist = np.array([np.convolve(r, k, "same") for r in hist]) + 1e-5
    hist /= hist.sum(1, keepdims=True); T = T / T.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(hist), np.log(T + 1e-9) + np.where(np.eye(3) > 0, -1e9, 0)

def semi_markov(E, logdur, logT, alpha, beta=1.0):
    """E (n,3) per-frame log-probabilities. Best segmentation under alpha*sum(E) + beta*(log p(len) + log T)."""
    n = len(E); C = np.vstack([np.zeros(3), np.cumsum(alpha * E, 0)])
    Vt = np.full((n + 1, 3), -np.inf); Vt[0] = 0
    M = np.full((n + 1, 3), -np.inf); M[0] = 0          # best predecessor score for a segment of state s starting here
    bd = np.zeros((n + 1, 3), np.int32); bs = np.zeros((n + 1, 3), np.int8)
    ld = beta * logdur
    for t in range(1, n + 1):
        lo = max(0, t - DMAX)
        for s in range(3):
            cand = M[lo:t, s] - C[lo:t, s] + ld[s, t - lo:0:-1]
            j = int(cand.argmax()); Vt[t, s] = cand[j] + C[t, s]; bd[t, s] = t - (lo + j)
        tr = Vt[t][:, None] + beta * logT                 # from s' (rows) to s (cols)
        tr[np.arange(3), np.arange(3)] = Vt[t] - 2.0      # a run longer than DMAX continues, at a small cost
        bs[t] = tr.argmax(0); M[t] = tr.max(0)
    out = np.zeros(n, np.int8); t = n; s = int(Vt[n].argmax())
    while t > 0:
        d = bd[t, s]; out[t - d:t] = s
        t -= d
        if t > 0: s = int(bs[t, s])
    return out

def emis(hl, dl):
    ls = lambda z: -np.logaddexp(0, -z)
    return np.c_[ls(-dl) + ls(hl), ls(-dl) + ls(-hl), ls(dl)]

def decode_sm(m, hl, dl, logdur, logT, alpha, beta):
    labp = np.full(len(m["labels25"]), "X", dtype="<U1")
    for h, (a, b) in enumerate(m["halves"]):
        s = m["half"] == h
        o = np.array(list(S))[semi_markov(emis(np.nan_to_num(hl[s]), np.nan_to_num(dl[s])), logdur, logT, alpha, beta)]
        labp[a:b + 1] = np.repeat(o, feats.STEP)[:b + 1 - a]
    return labp

GRID = [(a, b) for a in (1.0, 1.5, 2.5, 4.0) for b in (0.1, 0.25, 0.5)]
FW = {t: fit_fold(ms, Xw, t, "who", True) for t in PROVIDERS}
FD = {t: fit_fold(ms, Xd, t, "dead", False) for t in PROVIDERS}
DUR = {t: durations(t) for t in PROVIDERS}
def one(args):
    test, i = args; m = ms[i]
    hl, dl = apply(FW[test], Xw[i]), apply(FD[test], Xd[i])
    return np.array([score(m, decode_sm(m, hl, dl, *DUR[test], a, b)) for a, b in GRID])

def heldout_table(tag, res, grid):
    outs = []
    for t in PROVIDERS:
        tr = np.mean([np.mean([res[(t, i)][:, 0] for i in range(len(ms)) if ms[i]["prov"] == p], 0) for p in PROVIDERS if p != t], 0)
        te = np.mean([res[(t, i)] for i in range(len(ms)) if ms[i]["prov"] == t], 0)
        k = int(tr.argmax()); outs.append(te[k]); print(f"  {tag} {t}: picked {grid[k]} held-out {te[k][0]:.3f} best here {te[:,0].max():.3f}")
    o = np.mean(outs, 0); print(f"{tag}: spellF1 {o[0]:.3f} ({' '.join(f'{x[0]:.3f}' for x in outs)}) macro {o[1]:.3f} a/d {o[2]:.3f} team {o[3]:.3f}", flush=True)

if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        jobs = [(t, i) for t in PROVIDERS for i in range(len(ms))]
        res = dict(zip(jobs, ex.map(one, jobs)))
    heldout_table("semi-Markov", res, GRID)
