"""Fast decoder sweeps over prepared matches. eval_config(cfg) -> {set: mean folded spellF1}."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external")]
import hmm.decode as dec
from hmm.model import build_log_emit, rate_log_trans, viterbi, STATES
import possession_metrics as pm

PREP = os.path.join(ROOT, "logs_tune", "prep")
CACHE = os.path.join(ROOT, "data", "cache")
SETS = ("idsse", "metrica", "skillcorner")
DEFAULT = dict(weight=0.2, width_weight=0.05, cw=0.2, poss=12.0, stop=8.0, to_dead=0.4,
               loso="control_loso.npz", tabs=None, minposs=0.0, smooth=0, w_gap=1.0, w_place=1.0,
               w_speed=1.0, w_flight=1.0, cw_dead=None, clip=None)
_M, _T, _L = {}, None, {}

def names():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(PREP, "*_*.pkl")))

def match(n):
    if n not in _M:
        _M[n] = pickle.load(open(os.path.join(PREP, n + ".pkl"), "rb"))
    return _M[n]

def tables():
    global _T
    if _T is None:
        _T = pickle.load(open(os.path.join(PREP, "tables.pkl"), "rb"))
    return _T

def loso(f):
    if f not in _L:
        _L[f] = np.load(os.path.join(CACHE, f))
    return _L[f]

def smooth(x, half):
    if not half:
        return x
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    i = np.arange(len(x)); lo = np.clip(i - half, 0, len(x)); hi = np.clip(i + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]

def ratio_of(n, c):
    files = c["loso"] if isinstance(c["loso"], (list, tuple)) else [c["loso"]]
    p = np.mean([loso(f)[n] for f in files], axis=0)
    prior = np.mean([loso(f)["prior_" + n] for f in files], axis=0)
    p = smooth(p.astype(float), c["smooth"])
    r = np.log(np.clip(p, 1e-6, 1)) - np.log(prior)
    if c["clip"]:
        r = np.clip(r, -c["clip"], c["clip"])
    w = np.full(3, c["cw"])
    if c["cw_dead"] is not None:
        w[2] = c["cw_dead"]
    return r * w

_R = {}
def alt_tables(k, view, scheme):
    """tables for held-out set k from per-match rows: scheme 'pair' (idsse<->metrica, skillcorner
    from both) or 'all' (the other two providers), each provider counting once."""
    key = (k, view, scheme)
    if key not in _R:
        rows = _R.setdefault("rows", pickle.load(open(os.path.join(PREP, "rows.pkl"), "rb")))
        others = [o for o in SETS if o != k]
        if scheme == "pair" and k != "skillcorner":
            others = [o for o in others if o != "skillcorner"]
        def prov(o, what):
            return np.mean([np.array(r[what]) for (n, v), r in rows.items() if v == view and n.startswith(o)], axis=0)
        def mean(what):
            return np.mean([prov(o, what) for o in others], axis=0)
        sp = mean("speed"); fl = mean("flight"); gp = mean("gap"); wd = mean("width") + 1e-3
        wd = wd / wd.sum(axis=1, keepdims=True)
        _R[key] = [dec.speed_table(*sp), dec.flight_table(fl / fl.sum()),
                   dec.gap_distance_table(gp[0] / gp[0].sum(), gp[1] / gp[1].sum()), dec.width_table(*wd)]
    return _R[key]

def decode(n, c):
    m = match(n)
    speed_t, flight_t, gap_t, width_t = alt_tables(n.split("_")[0], *c["tabs"]) if c["tabs"] else tables()[n.split("_")[0]]
    W = c["weight"]
    clues = [(m["gapdist"], gap_t, W * c["w_gap"]), (m["place"], dec.place_table(), W * c["w_place"]),
             (m["speed"], speed_t, W * c["w_speed"]), (m["flight"], flight_t, W * c["w_flight"]),
             (m["width"], width_t, c["width_weight"])]
    fps = m["fps"]
    trans = rate_log_trans(c["poss"] * fps, c["stop"] * fps, c["to_dead"])
    start = np.log(dec.START)
    direct = ratio_of(n, c) if c["cw"] or c["cw_dead"] else None
    pred = np.full(len(m["labels"]), "X", dtype="<U1")
    fold = int(round(c["minposs"] * fps))
    for lo, hi in m["segments"]:
        le = build_log_emit([(o[lo:hi], t, w) for o, t, w in clues])
        if direct is not None:
            le = le + direct[lo:hi]
        pred[lo:hi] = STATES[dec.fold_brief_possessions(viterbi(le, trans, start), fold)]
    pred[m["labels"] == "X"] = "X"
    return pred

def spell(gt, pred):
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    p = pm.fill_abstentions(p, runs)
    return pm.spell_scores(g, p, runs)["mean_f1"]

def _job(a):
    n, c = a
    m = match(n)
    pred = decode(n, c)
    return n, spell(m["folded"], pred), spell(m["labels"], pred)

_POOL = None
def eval_configs(cfgs, sets=SETS, jobs=4):
    """list of {set: folded mean} (+ 'mean', 'raw'), one per config"""
    global _POOL
    if _POOL is None:
        _POOL = Pool(jobs)
    ns = [n for n in names() if n.startswith(tuple(sets))]
    full = [dict(DEFAULT, **c) for c in cfgs]
    res = _POOL.map(_job, [(n, c) for c in full for n in ns], chunksize=1)
    out = []
    for i in range(len(full)):
        r = res[i * len(ns):(i + 1) * len(ns)]
        o = {k: float(np.mean([f for n, f, _ in r if n.startswith(k)])) for k in sets}
        o["mean"] = float(np.mean([o[k] for k in sets]))
        o["raw"] = float(np.mean([np.mean([x for n, _, x in r if n.startswith(k)]) for k in sets]))
        o["per"] = {n: f for n, f, _ in r}
        out.append(o)
    return out

def show(tag, o):
    print("{:40s} {:.3f} / {:.3f} / {:.3f}  mean {:.4f}  raw {:.3f}".format(
        tag, o["idsse"], o["metrica"], o["skillcorner"], o["mean"], o["raw"]), flush=True)
