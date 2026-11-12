"""Decode comparison on the best linear scores: fixed minimum runs vs a two-chain Viterbi."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
Xw, _ = design(ms, "who", list(ms[0]["who"]), ["s1", "s3", "d2"])
names = list(ms[0]["dead"])
B = design(ms, "dead", ["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])[0]
R = design(ms, "dead", [n for n in names if n not in ("sp_med", "acc_p80")], ["s1", "s3", "d2"])[0]
Xd = [np.c_[a, b] for a, b in zip(B, R)]

def viterbi2(logit, cost01, cost10):
    """Two-state path maximising sum of +-logit/2 minus switch costs. Returns bool (state 1)."""
    n = len(logit); e = np.c_[-logit / 2, logit / 2]
    back = np.zeros((n, 2), np.int8); s = e[0].copy()
    for t in range(1, n):
        a0 = (s[0], s[1] - cost10); a1 = (s[0] - cost01, s[1])
        back[t, 0] = a0[1] > a0[0]; back[t, 1] = a1[1] > a1[0]
        s = np.array([max(a0) + e[t, 0], max(a1) + e[t, 1]])
    out = np.zeros(n, bool); k = int(s[1] > s[0])
    for t in range(n - 1, -1, -1):
        out[t] = k; k = back[t, k]
    return out

def decode_v(m, hl, dl, cw, cd):
    labp = np.full(len(m["labels25"]), "X", dtype="<U1")
    for h, (a, b) in enumerate(m["halves"]):
        s = m["half"] == h
        home = viterbi2(np.nan_to_num(hl[s]), cw, cw)
        dead = viterbi2(np.nan_to_num(dl[s]), cd, cd)
        o = np.where(dead, "D", np.where(home, "H", "A"))
        labp[a:b + 1] = np.repeat(o, feats.STEP)[:b + 1 - a]
    return labp

CW = [2, 4, 6, 9, 13]; CD = [2, 4, 6, 9, 13]
MT = [(1, 2, 1), (2, 3, 2), (2, 5, 2), (3, 3, 2), (3, 5, 3), (1.5, 3, 1.5)]
from concurrent.futures import ProcessPoolExecutor
def one(args):
    test, i = args
    m = ms[i]
    hl, dl = apply(FW[test], Xw[i]), apply(FD[test], Xd[i])
    v = [[score(m, decode_v(m, hl, dl, cw, cd))[0] for cd in CD] for cw in CW]
    a = [score(m, decode(m, hl, dl, mt, md, ma))[0] for mt, md, ma in MT]
    return np.array(v), np.array(a)
FW = {t: fit_fold(ms, Xw, t, "who", True) for t in PROVIDERS}
FD = {t: fit_fold(ms, Xd, t, "dead", False) for t in PROVIDERS}
if __name__ == "__main__":
    # scores of every match under each fold's fit (needed to pick settings on the other providers)
    res = {}
    with ProcessPoolExecutor(4) as ex:
        jobs = [(t, i) for t in PROVIDERS for i in range(len(ms))]
        for j, r in zip(jobs, ex.map(one, jobs)): res[j] = r
    for kind, k in (("viterbi", 0), ("absorb", 1)):
        outs = []
        for t in PROVIDERS:
            tr = np.mean([np.mean([res[(t, i)][k] for i in range(len(ms)) if ms[i]["prov"] == p], 0) for p in PROVIDERS if p != t], 0)
            te = np.mean([res[(t, i)][k] for i in range(len(ms)) if ms[i]["prov"] == t], 0)
            pick = np.unravel_index(tr.argmax(), tr.shape)
            outs.append(te[pick]); print(kind, t, "picked", pick, f"held-out {te[pick]:.3f}  best here {te.max():.3f}")
        print(kind, "mean", round(float(np.mean(outs)), 3))
