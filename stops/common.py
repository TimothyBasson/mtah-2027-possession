"""Shared loading and scoring for stops/: the held-out probabilities of a deep/ tag, the outside
truth as given and folded to the control rule, and spellF1 per match."""

import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)
PROVIDERS = ("idsse", "metrica", "skillcorner")
CACHE = os.path.join(ROOT, "data", "cache", "stops")


def paths():
    return sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))


def _truth(path):
    import score_hmm as sh
    from control import control_labels
    d = sh.load(path)
    return os.path.basename(path)[:-4], dict(raw=np.array(d["labels"]), fold=control_labels(d),
                                             spans=[tuple(int(v) for v in h) for h in d["halves"]])


def truths():
    """{match: {raw, fold, spans}}; cached, the folded labels take a minute to build."""
    f = os.path.join(CACHE, "truth.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    with Pool(4) as pool:
        out = dict(pool.map(_truth, paths()))
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def probs(tag, name, fold=None):
    """Held-out (or a given fold's) per-half probabilities at 5 fps."""
    p = np.load(os.path.join(ROOT, "data", "cache", "deep", tag, fold or name.split("_")[0], name + ".npz"))
    return [p[k] for k in sorted(p.files) if k.startswith("p")]


def spell(truth, pred):
    pred = np.array(pred, copy=True)
    pred[truth == "X"] = "X"
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"]


def table(rows, names, cols):
    """rows (matches, cols) -> provider means and the equal-provider mean, printed."""
    rows = np.asarray(rows)
    out = []
    for k in PROVIDERS:
        m = np.array([n.startswith(k) for n in names])
        out.append(rows[m].mean(0))
    out.append(np.mean(out, 0))
    print(f"{'':14s}" + "".join(f"{c:>12s}" for c in cols))
    for k, r in zip(PROVIDERS + ("mean",), out):
        print(f"{k:14s}" + "".join(f"{x:12.3f}" for x in r))
    return np.array(out)
