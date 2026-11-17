"""Per keeper note on BAR-ATH: seconds of D in the 10 s after the note, v9 against each variant;
and every place a variant changes v9, matched to the notes.  python kp/notes.py box4 box8 ..."""
import csv, json, sys
import numpy as np
def L(p): return np.array([r["label"] for r in csv.DictReader(open(p))])
base = L("evaluation/results/kp_base.csv")
notes = json.load(open("data/viz/notes/BAR-ATH.json"))["notes"]
KEEP = [23946, 26688, 69326, 76462, 82248, 123481, 133928, 168311]
for a in sys.argv[1:]:
    new = L("evaluation/results/kp_%s.csv" % a)
    g = base != "X"
    print("%s: %.2f%% of graded frames differ; D share %.3f -> %.3f" % (
        a, 100 * (base != new)[g].mean(), (base[g] == "D").mean(), (new[g] == "D").mean()))
    for n in notes:
        f = n["frame"]
        if n["tag"] == "False stoppage" or "keeper" in n["text"].lower():
            w = slice(f - 25, f + 250)
            lab = lambda x: "".join(k for k, _ in __import__("itertools").groupby(x[w]))
            print("   %6d %8s %s %-15s D next 10 s: %4.1f -> %4.1f   %s -> %s" % (
                f, "%d:%04.1f" % divmod(f / 25, 60), "K" if f in KEEP else " ", n["tag"][:15],
                (base[w] == "D").sum() / 25, (new[w] == "D").sum() / 25, lab(base), lab(new)))
    idx = np.where(base != new)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 50)[0] + 1) if len(idx) else []
    for gr in groups:
        lo = gr[0]
        hit = [n["frame"] for n in notes if lo - 250 <= n["frame"] <= gr[-1] + 250]
        print("   change %6d %8s %5.1f s %-12s notes %s" % (lo, "%d:%04.1f" % divmod(lo / 25, 60),
              len(gr) / 25, ",".join(sorted({base[i] + ">" + new[i] for i in gr})), hit))
