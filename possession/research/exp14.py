"""How strongly, and how late, do the players answer a turnover? Mean of each signal around true
open-play turnovers (to home minus to away, halved), in units of the signal's own sd."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths):
    m["who"].update(feats2.cached(p)); z = np.load(os.path.join(ROOT, "data", "cache", "research4", os.path.basename(p))); m["who"].update({k: z[k] for k in z})
lags = np.arange(-3 * HZ, 6 * HZ + 1)
show = [-3, -2, -1, -0.4, 0, 0.4, 1, 1.6, 2, 3, 4, 6]
print("lag s:".ljust(26) + " ".join(f"{s:+5.1f}" for s in show))
for nm in ["acc_x", "acc_x_top4", "reversals", "sprint_onset", "vcx", "press_max", "speed", "wid", "len"]:
    for prov in PROVIDERS:
        acc = np.zeros(len(lags)); cnt = 0; sds = []
        for m in ms:
            if m["prov"] != prov: continue
            x = m["who"][nm]; lab_ = m["labels"]; sds.append(np.nanstd(x))
            t = np.flatnonzero((lab_[1:] != lab_[:-1]) & np.isin(lab_[1:], ["H", "A"]) & np.isin(lab_[:-1], ["H", "A"])) + 1
            t = t[(t + lags[0] >= 0) & (t + lags[-1] < len(x)) & (m["half"][np.clip(t + lags[0], 0, len(x) - 1)] == m["half"][np.clip(t + lags[-1], 0, len(x) - 1)])]
            sgn = np.where(lab_[t] == "H", 1.0, -1.0)
            acc += np.nansum(x[t[:, None] + lags[None]] * sgn[:, None], 0); cnt += len(t)
        prof = acc / cnt / np.mean(sds)
        print(f"{nm:14s}{prov:12s}" + " ".join(f"{prof[np.argmin(np.abs(lags / HZ - s))]:+5.2f}" for s in show))
