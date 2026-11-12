"""Compact sets, paired by match against the three-feature team model."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
V = ["s1", "s3", "d2"]
BD = (["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])

def per_match(who, dead_blocks):
    Xw, _ = design(ms, "who", *who)
    Xd = None
    for names, views in dead_blocks:
        X, _ = design(ms, "dead", names, views)
        Xd = X if Xd is None else [np.c_[a, b] for a, b in zip(Xd, X)]
    out = np.zeros((len(ms), 4))
    for test in PROVIDERS:
        fw = fit_fold(ms, Xw, test, "who", True); fd = fit_fold(ms, Xd, test, "dead", False)
        for i, m in enumerate(ms):
            if m["prov"] == test:
                out[i] = score(m, decode(m, apply(fw, Xw[i]), apply(fd, Xd[i])))
    return out

def show(tag, s, ref=None):
    pm_ = [s[[i for i, m in enumerate(ms) if m["prov"] == p]].mean(0) for p in PROVIDERS]
    mean = np.mean(pm_, 0)
    line = f"{tag:46s} spellF1 {mean[0]:.3f} ({' '.join(f'{x[0]:.3f}' for x in pm_)}) macro {mean[1]:.3f} a/d {mean[2]:.3f} team {mean[3]:.3f}"
    if ref is not None:
        d = s[:, 0] - ref[:, 0]
        line += f" | paired {d.mean():+.3f} se {d.std(ddof=1)/np.sqrt(len(d)):.3f} up {int((d>0).sum())}/30 worst {d.min():+.3f}"
    print(line, flush=True)

base = ["wid", "len", "vcx"]
allw = list(ms[0]["who"]); alld = [n for n in ms[0]["dead"] if n not in ("sp_med", "acc_p80")]
ref = per_match((base, V), [BD]); show("3 who + 3 dead (reference)", ref)
sets = {
 "who6: +speed press_max pol": base + ["speed", "press_max", "pol"],
 "who8: + nn_opp_back vor": base + ["speed", "press_max", "pol", "nn_opp_back", "vor"],
 "who10: + speed_sd keeper_speed": base + ["speed", "press_max", "pol", "nn_opp_back", "vor", "speed_sd", "keeper_speed"],
 "who all 42": allw,
}
keep = {}
for k, v in sets.items():
    keep[k] = per_match((v, V), [BD]); show(k, keep[k], ref)
dsets = {
 "dead +corner_min box_max": ["corner_min", "box_max"],
 "dead +corner box spread_sum": ["corner_min", "box_max", "spread_sum"],
 "dead +corner box spread_sum own_half sp_p80": ["corner_min", "box_max", "spread_sum", "own_half", "sp_p80"],
 "dead all 15": alld,
}
w8 = sets["who8: + nn_opp_back vor"]
for k, v in dsets.items():
    show("who8, " + k, per_match((w8, V), [BD, (v, V)]), ref)
show("who all, dead all", per_match((allw, V), [BD, (alld, V)]), ref)
