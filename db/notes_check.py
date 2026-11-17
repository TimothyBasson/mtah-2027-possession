import json, csv, sys
import numpy as np
def L(p): return np.array([r["label"] for r in csv.DictReader(open(p))])
base = L("evaluation/results/db_ahead0.csv")
notes = json.load(open("data/viz/notes/BAR-ATH.json"))["notes"]
for a in sys.argv[1:]:
    new = L("evaluation/results/db_ahead%s.csv" % a)
    diff = base != new
    g = (base != "X")
    print("ahead x%s: %.2f%% of graded frames differ; D share %.3f -> %.3f" % (a, 100*diff[g].mean(), (base[g]=="D").mean(), (new[g]=="D").mean()))
    for n in notes:
        f = n["frame"]; w = slice(f - 25, f + 250)  # the 11 s after the note
        if n["tag"] in ("False stoppage", "Missed stoppage") or "keeper" in n["text"].lower():
            print("   %6d %-16s D in next 10 s: v9 %4.1fs -> %4.1fs" % (f, n["tag"], (base[w]=="D").sum()/25, (new[w]=="D").sum()/25))
    # changed stretches
    idx = np.where(diff)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
    print("   changed stretches:", len(runs), [ (int(r[0]), round(len(r)/25,1), base[r[0]]+">"+new[r[0]]) for r in runs])
