"""Probability-averaged sets of tags, held out, raw and folded labels, shipped decode.
    python deep/short/ens.py bt,bts1 bt,bts1,pp sw25
A tag with no net for a fold (sw25 and sw0 have no skillcorner fold: nothing differs there) falls
back to bt for that fold."""
import glob, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode, score_hmm as sh
from control import control_labels
from run import DECODE
DECODE = dict(DECODE, **{k: float(v) for k, v in (kv.split("=") for kv in os.environ.get("DEC", "").split(",") if kv)})  # DEC=min_team=2,cost=8

def one(args):
    path, sets = args
    name = os.path.basename(path)[:-4]; prov = name.split("_")[0]
    d = sh.load(path); spans = [tuple(h) for h in d["halves"]]
    truths = (d["labels"], control_labels(d)); out = []
    for tags in sets:
        ps = []
        for t in tags:
            f = os.path.join(ROOT, "data", "cache", "deep", t, prov, name + ".npz")
            ps.append(np.load(f if os.path.exists(f) else f.replace(os.sep + t + os.sep, os.sep + "bt" + os.sep)))
        probs = []
        for i, (a, b) in enumerate(spans):
            probs.append(np.mean([decode.upsample(p[f"p{i}"], b - a + 1) for p in ps], 0)[::5])
        pred = decode.labels_from(probs, spans, len(d["labels"]), **DECODE)
        pred[d["labels"] == "X"] = "X"
        out.append([sh.score(gt, pred)[0] for gt in truths])
    return name, np.array(out)

if __name__ == "__main__":
    sets = [s.split(",") for s in sys.argv[1:]]
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    with Pool(2) as pool: rows = pool.map(one, [(p, sets) for p in paths])
    names = [n for n, _ in rows]; S = np.stack([r for _, r in rows])
    print(f"{'set':22s}  raw: idsse metrica skillc  mean  | folded: idsse metrica skillc  mean")
    for si, s in enumerate(sets):
        line = f"{','.join(s):22s} "
        for ti in (0, 1):
            m = [S[[n.startswith(p) for n in names], si, ti].mean() for p in ("idsse", "metrica", "skillcorner")]
            line += "      " + "  ".join(f"{x:.3f}" for x in m) + f"  {np.mean(m):.3f}  |"
        print(line)
