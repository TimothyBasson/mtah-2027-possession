"""Every v9 dead spell on BAR-ATH: how the ball disappears and reappears around it."""
import json, os, sys, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, "data/cache/barath_pos.npz")); px, py, bx, by = z["px"], z["py"], z["bx"], z["by"]
KP = (23, 45); L, W = 105.3, 68.0
lab = np.array([r["label"] for r in csv.DictReader(open("/mnt/project-files/v9/hmm_v9.csv"))])
notes = json.load(open(os.path.join(ROOT, "data/viz/notes/BAR-ATH.json")))["notes"]
seen = np.isfinite(bx) & (np.maximum.reduce([-bx, bx - L, -by, by - W]) <= 5)
def kd(g):
    return min(np.nan_to_num(np.hypot(px[g, k] - bx[g], py[g, k] - by[g]), nan=99) for k in KP)
def out(g):
    return max(-bx[g], bx[g] - L, -by[g], by[g] - W)
runs = []; s = 0
for i in range(1, len(lab) + 1):
    if i == len(lab) or lab[i] != lab[s]:
        runs.append((s, i, lab[s])); s = i
for a, b, l in runs:
    if l != "D" or b - a < 25: continue
    tag = [n["tag"] for n in notes if a - 50 <= n["frame"] <= b and n["tag"] in ("False stoppage",)]
    # last sighting before the spell start (or at it)
    prev = np.where(seen[max(0, a - 750):a + 1])[0]
    p = max(0, a - 750) + prev[-1] if len(prev) else None
    sa = seen[a:b].mean()
    nxt = np.where(seen[a:b + 750])[0]
    q = a + nxt[0] if len(nxt) else None
    # first sighting after a gap that starts in the spell
    gap_end = None
    if p is not None:
        j = p + 1
        while j < len(seen) and not seen[j]: j += 1
        gap_end = j
    def desc(g):
        if g is None or g >= len(bx): return "-"
        return "(%3.0f,%2.0f) out%+5.1f kd%4.1f" % (bx[g], by[g], out(g), kd(g))
    print("%6d %5.1fs seen%3.0f%% | last %s @%+.1fs | back %s @%+.1fs %s" % (
        a, (b - a) / 25, 100 * sa, desc(p), ((p - a) / 25 if p is not None else 0),
        desc(gap_end), ((gap_end - a) / 25 if gap_end is not None else 0), "<<FALSE" if tag else ""))
