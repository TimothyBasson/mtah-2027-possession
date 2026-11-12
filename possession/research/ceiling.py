import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
ms = load()
V = ["s1", "s3", "d2"]
names = list(ms[0]["who"])
Xw, _ = design(ms, "who", names, V)
Xd, _ = design(ms, "dead", ["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])
res = {k: [] for k in ("pred", "true_dead", "true_who", "true_both5hz")}
bins = np.array([0, 1, 2, 3, 5, 8, 12, 20, 1e9]); hit = np.zeros((3, len(bins) - 1)); cnt = np.zeros((3, len(bins) - 1))
lhit = np.zeros((3, len(bins) - 1)); lcnt = np.zeros((3, len(bins) - 1))
for pi, test in enumerate(PROVIDERS):
    fw = fit_fold(ms, Xw, test, "who", True); fd = fit_fold(ms, Xd, test, "dead", False)
    out = {k: [] for k in res}
    for m, xw, xd in zip(ms, Xw, Xd):
        if m["prov"] != test: continue
        hl, dl = apply(fw, xw), apply(fd, xd)
        lab = m["labels"]
        tdead = np.where(lab == "D", 5.0, -5.0)
        # true who: carry the true team through dead frames
        tw = np.where(lab == "H", 5.0, np.where(lab == "A", -5.0, np.nan))
        i = np.arange(len(tw)); ok = np.isfinite(tw); last = np.maximum.accumulate(np.where(ok, i, 0)); tw = np.nan_to_num(tw[last])
        out["pred"].append(score(m, decode(m, hl, dl)))
        out["true_dead"].append(score(m, decode(m, hl, tdead, min_dead=0, min_alive=0)))
        out["true_who"].append(score(m, decode(m, tw, dl, min_team=0)))
        out["true_both5hz"].append(score(m, decode(m, tw, tdead, 0, 0, 0)))
        # who accuracy by distance to the nearest true change of label, and by true spell length
        live = np.isin(lab, ["H", "A"])
        ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        edges = np.r_[0, ch, len(lab)]
        seg = np.searchsorted(edges, np.arange(len(lab)), "right") - 1
        dist = np.minimum(np.arange(len(lab)) - edges[seg], edges[seg + 1] - 1 - np.arange(len(lab))) / HZ
        length = (edges[seg + 1] - edges[seg]) / HZ
        ok = (hl >= 0) == (lab == "H")
        for arr, H, C in ((dist, hit, cnt), (length, lhit, lcnt)):
            b = np.digitize(arr[live], bins) - 1
            np.add.at(H[pi], b, ok[live]); np.add.at(C[pi], b, 1)
    for k in res: res[k].append(np.mean(out[k], 0))
for k in res:
    a = np.array(res[k]); print(f"{k:14s} spellF1 {a[:,0].mean():.3f} ({a[:,0].round(3)})  macro {a[:,1].mean():.3f} a/d {a[:,2].mean():.3f} team {a[:,3].mean():.3f}")
print("who accuracy by seconds from the nearest true boundary:", bins[:-1])
print((hit / cnt).round(3)); print("share of live frames", (cnt / cnt.sum(1, keepdims=True)).round(3))
print("who accuracy by true spell length (s):")
print((lhit / lcnt).round(3)); print("share", (lcnt / lcnt.sum(1, keepdims=True)).round(3))
