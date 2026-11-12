import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
ms = load()
names = list(ms[0]["dead"])
B = design(ms, "dead", ["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])[0]
R = design(ms, "dead", [n for n in names if n not in ("sp_med", "acc_p80")], ["s1", "s3", "d2"])[0]
Xd = [np.c_[a, b] for a, b in zip(B, R)]
bins = np.array([0, 2, 4, 6, 10, 15, 25, 40, 1e9])
for test in PROVIDERS:
    fd = fit_fold(ms, Xd, test, "dead", False)
    rec = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1); fr = np.zeros(len(bins) - 1)
    prec = np.zeros(len(bins) - 1); pcnt = np.zeros(len(bins) - 1); pfr = np.zeros(len(bins)-1)
    for m, xd in zip(ms, Xd):
        if m["prov"] != test: continue
        dl = apply(fd, xd); lab = m["labels"]
        for h in (0, 1):
            s = m["half"] == h
            pd_ = ~absorb(np.nan_to_num(dl[s]) < 0, {True: 2 * HZ, False: 3 * HZ})
            td = lab[s] == "D"; gr = lab[s] != "X"
            for truth, pred, R_, C_, F_ in ((td, pd_, rec, cnt, fr), (pd_, td, prec, pcnt, pfr)):
                i = 0
                for k, g in itertools.groupby(truth):
                    n = len(list(g))
                    if k and gr[i:i+n].all():
                        b = np.digitize(n / HZ, bins) - 1
                        R_[b] += pred[i:i + n].mean(); C_[b] += 1; F_[b] += n
                    i += n
    print(test, "true D runs by length (s)", bins[:-1].astype(int))
    print("  count/match", (cnt / sum(m['prov']==test for m in ms)).round(1), " frame share", (fr / fr.sum()).round(2))
    print("  mean fraction called dead", (rec / cnt).round(2))
    print("  predicted D runs: count/match", (pcnt / sum(m['prov']==test for m in ms)).round(1), "frame share", (pfr/pfr.sum()).round(2), " truly dead frac", (prec / pcnt).round(2))
