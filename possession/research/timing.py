import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
Xw, _ = design(ms, "who", list(ms[0]["who"]), ["s1", "s3", "d2"])
for test in PROVIDERS:
    fw = fit_fold(ms, Xw, test, "who", True)
    offs, miss, tot, npred, held = [], 0, 0, 0, []
    for m, xw in zip(ms, Xw):
        if m["prov"] != test: continue
        hl = apply(fw, xw); lab = m["labels"]
        for h in (0, 1):
            s = m["half"] == h
            pr = absorb(np.nan_to_num(hl[s]) >= 0, {True: 2 * HZ, False: 2 * HZ})
            l = lab[s]
            t = np.flatnonzero((l[1:] != l[:-1]) & np.isin(l[1:], ["H", "A"]) & np.isin(l[:-1], ["H", "A"])) + 1
            pc = np.flatnonzero(pr[1:] != pr[:-1]) + 1
            pdir = pr[pc]
            npred += len(pc)
            for ti in t:
                tot += 1
                c = pc[(pdir == (l[ti] == "H"))]
                # spell length after the turnover
                if len(c) and np.abs(c - ti).min() <= 5 * HZ:
                    offs.append((c[np.abs(c - ti).argmin()] - ti) / HZ)
                else:
                    miss += 1
    o = np.array(offs)
    print(f"{test:12s} true open-play turnovers {tot}, predicted lean changes {npred}, unmatched within 5 s {miss/tot:.2f}; "
          f"offset s: mean {o.mean():+.2f} median {np.median(o):+.2f} sd {o.std():.2f} |o|<=0.5: {(np.abs(o)<=0.5).mean():.2f} |o|<=1: {(np.abs(o)<=1).mean():.2f} |o|<=2: {(np.abs(o)<=2).mean():.2f}")
