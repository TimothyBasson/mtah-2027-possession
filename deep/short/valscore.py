"""spellF1 of tags on each fold's validation matches (unseen matches of the TRAINING providers)
and on its held-out provider, raw and folded labels, shipped decode.
    python deep/short/valscore.py bt tb"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode, score_hmm as sh
from control import control_labels
from run import DECODE

def one(f):
    z = np.load(f); name = os.path.basename(f)[:-4]
    if str(z["kind"]) == "train": return None
    d = sh.load(os.path.join(ROOT, "data", "external", name + ".npz"))
    spans = [tuple(h) for h in d["halves"]]
    pred = decode.labels_from([z[f"p{i}"] for i in range(len(spans))], spans, len(d["labels"]), **DECODE)
    pred[d["labels"] == "X"] = "X"
    return f.split(os.sep)[-3], f.split(os.sep)[-2], name, str(z["kind"]), sh.score(d["labels"], pred)[0], sh.score(control_labels(d), pred)[0]

if __name__ == "__main__":
    fs = [f for t in sys.argv[1:] for f in sorted(glob.glob(os.path.join(ROOT, "data", "cache", "deep", t, "*", "*.npz")))]
    with Pool(2) as p: R = [r for r in p.map(one, fs) if r]
    for t in sys.argv[1:]:
        for fold in ("idsse", "metrica", "skillcorner"):
            line = f"{t:6s} fold {fold:12s}"
            for kind in ("val", "test"):
                for prov in ("idsse", "metrica", "skillcorner"):
                    x = [(r[4], r[5]) for r in R if r[0] == t and r[1] == fold and r[3] == kind and r[2].startswith(prov)]
                    if x: line += f"  {kind} {prov[:4]} {np.mean(x, 0)[0]:.3f}/{np.mean(x, 0)[1]:.3f} ({len(x)})"
            print(line)
