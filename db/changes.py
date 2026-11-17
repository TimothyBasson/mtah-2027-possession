"""Where --ahead changes v9 on BAR-ATH: stretches merged when under 2 s apart, matched to notes."""
import json, csv, sys
import numpy as np
def L(p): return np.array([r["label"] for r in csv.DictReader(open(p))])
base = L("evaluation/results/db_ahead0.csv")
notes = json.load(open("data/viz/notes/BAR-ATH.json"))["notes"]
for a in sys.argv[1:]:
    new = L("evaluation/results/db_ahead%s.csv" % a)
    idx = np.where(base != new)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 50)[0] + 1) if len(idx) else []
    print("x%s: %d frames differ (%.1f s, %.2f%% of graded), %d places" % (
        a, len(idx), len(idx) / 25, 100 * len(idx) / (base != "X").sum(), len(groups)))
    near_note = 0
    for g in groups:
        lo, hi = g[0], g[-1]
        kinds = sorted({base[i] + ">" + new[i] for i in g})
        hit = [n for n in notes if lo - 250 <= n["frame"] <= hi + 250]
        near_note += bool(hit)
        print("  %6d %s  %5.1f s changed  %-18s %s" % (lo, "%d:%04.1f" % divmod(lo / 25, 60), len(g) / 25,
              ",".join(kinds), "; ".join("note %d %s" % (n["frame"], n["tag"]) for n in hit)))
    print("  places within 10 s of a note: %d of %d" % (near_note, len(groups)))
