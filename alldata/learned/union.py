"""Of the true possessions a base run misses (IoU 0.5, labels as built), how many does another
family find?  python alldata/learned/union.py  (run from the repository root; sources as blend.py)"""
import sys, glob, os, numpy as np
sys.path.insert(0, "alldata/learned")
import blend as B, score as S
from score import pm, decode
tot = {}
for fold in ("idsse", "metrica", "skillcorner"):
    for f in sorted(glob.glob(os.path.join(S.tag_dir("pw1e"), fold, "*.npz"))):
        if str(np.load(f)["kind"]) != "test": continue
        name = os.path.basename(f)[:-4]
        raw, fl, spans = S.truths(name)
        runs = pm.valid_runs(raw); g = pm.possession_spells(raw, runs)
        fl_ = {}
        for tag in ("pw1e", "timlab", "phys:sLh", "deep:bt"):
            p = pm.label_codes(decode.labels_from(B.load(tag, fold, name, "", len(spans), spans), spans, len(raw), **S.RAW))
            pr = pm.possession_spells(p, runs)
            fl_[tag] = pm.matched_gt_flags(pm.spell_overlaps(g, pr), len(g), 0.5)
        b = fl_["pw1e"]
        t = tot.setdefault(fold, np.zeros(6))
        t += [len(g), b.sum(), (~b & fl_["timlab"]).sum(), (~b & fl_["phys:sLh"]).sum(), (~b & fl_["deep:bt"]).sum(), (~b & (fl_["timlab"] | fl_["phys:sLh"] | fl_["deep:bt"])).sum()]
for k, t in tot.items():
    m = t[0] - t[1]
    print(k, f"true {int(t[0])}, learned finds {t[1]/t[0]:.2f}; of its {int(m)} misses found by Tim {t[2]/m:.2f}, physics {t[3]/m:.2f}, players-only {t[4]/m:.2f}, any {t[5]/m:.2f}; union recall {(t[1]+t[5])/t[0]:.2f}")
