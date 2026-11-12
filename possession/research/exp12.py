"""The shipped model (possession/model.py) when players go missing as on BAR-ATH: weights fitted
on complete tracking against weights fitted on tracking with players knocked out, both scored on
held-out matches with players knocked out (and on complete ones).

    uv run python possession/research/exp12.py
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), HERE]
from score import frames, pm  # noqa: E402
from exp9 import knock  # noqa: E402
import zlib

def seed(path): return zlib.crc32(os.path.basename(path).encode())
def prov(path): return os.path.basename(path).split("_")[0]

def load(path, knocked):
    d = np.load(path)
    P = knock(d["players"], seed(path)) if knocked else d["players"]
    if knocked:  # speed column must go missing with the position
        P[~np.isfinite(P[:, :, 0]), 2] = np.nan
    return d, P

def fit_fold(args):
    train, knocked = args
    import model
    ms = []
    for p in train:
        d, P = load(p, knocked)
        ms.append((prov(p), P, d["labels"], [tuple(h) for h in d["halves"]], d["team"], d["pitch"]))
    return model.fit(ms)

def score_match(args):
    path, fits = args      # fits: {name: fitted}
    import features, model
    out = {}
    for knocked in (False, True):
        d, P = load(path, knocked)
        halves = [tuple(h) for h in d["halves"]]
        mask = np.zeros(len(d["labels"]), bool)
        for s, e in halves: mask[s:e + 1] = True
        features.PITCH = model.PITCH = tuple(d["pitch"]); model.HALVES = halves
        model.graded_mask = lambda n: mask
        F = frames(P, d["ball"], d["team"], features)
        truth = pm.label_codes(d["labels"])
        for name, f in fits.items():
            r = pm.evaluate(truth, pm.label_codes(model.predict(F, f)), bootstrap_n=0)
            out[(name, knocked)] = (r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"])
    return out

if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    sets = sorted({prov(p) for p in paths})
    res = {}
    with ProcessPoolExecutor(3) as ex:
        jobs = [([p for p in paths if prov(p) != s], k) for s in sets for k in (False, True)]
        fits = list(ex.map(fit_fold, jobs))
        F = {(s, k): f for (s, k), f in zip([(s, k) for s in sets for k in (False, True)], fits)}
        for s in sets:
            print(s, "complete fit:", F[(s, False)]); print(s, "knocked fit:", F[(s, True)], flush=True)
        tasks = [(p, {"complete": F[(prov(p), False)], "knocked": F[(prov(p), True)]}) for p in paths]
        outs = list(ex.map(score_match, tasks))
    for name in ("complete", "knocked"):
        for knocked in (False, True):
            rows = [np.mean([o[(name, knocked)] for p, o in zip(paths, outs) if prov(p) == s], 0) for s in sets]
            m = np.mean(rows, 0)
            print(f"weights from {name:8s} tracking, scored on {'knocked-out' if knocked else 'complete   '}: spellF1 {m[0]:.3f} ({' '.join(f'{r[0]:.3f}' for r in rows)}) macro {m[1]:.3f} a/d {m[2]:.3f} team {m[3]:.3f}", flush=True)
    d = np.array([o[("knocked", True)][0] - o[("complete", True)][0] for o in outs])
    print(f"paired on knocked-out matches, knocked fit minus complete fit: {d.mean():+.4f} se {d.std(ddof=1)/np.sqrt(len(d)):.4f} up {(d>0).sum()}/30")
