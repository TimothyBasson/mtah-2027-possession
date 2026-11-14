import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, ROOT + "/evaluation", ROOT + "/external"]
import hmm.decode as dec
import possession_metrics as pm
from hmm.model import viterbi, STATES
from multiprocessing import Pool
C = ROOT + "/data/cache/wip/emit"
NAMES = sorted(os.path.basename(p)[:-4] for p in glob.glob(C + "/*.npz"))
SETS = ("idsse", "metrica", "skillcorner")
V7 = dict(idsse=0.609, metrica=0.798, skillcorner=0.428)
def load(n): return dict(np.load(f"{C}/{n}.npz"))
def loso(f="control_loso.npz"): return dict(np.load(f"{ROOT}/data/cache/{f}"))
def ratio(L, n): return np.log(np.clip(L[n], 1e-6, 1)) - np.log(L["prior_" + n])
def decode(m, direct=None, trans=None, post=None):
    trans = dec.transitions() if trans is None else trans
    pred = np.full(len(m["labels"]), "X", "<U1")
    for a, b in m["halves"]:
        e = m["emit"][a:b+1].astype(float)
        if direct is not None: e = e + direct[a:b+1]
        p = STATES[viterbi(e, trans, np.log(dec.START))]
        if post: p = post(p)
        pred[a:b+1] = p
    pred[m["labels"] == "X"] = "X"
    return pred
def sf1(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]])
def report(res, tag=""):
    """res: {name: (4,) folded scores}"""
    out = []
    for k in SETS:
        v = np.mean([res[n] for n in res if n.startswith(k)], 0); out.append(v)
    m = np.mean(out, 0)
    flag = "".join("!" if out[i][0] < V7[k] - 0.01 else "" for i, k in enumerate(SETS))
    print(f"{tag:38s} " + " ".join(f"{o[0]:.3f}" for o in out) + f" | mean {m[0]:.3f} {flag} | a/d " +
          " ".join(f"{o[2]:.3f}" for o in out) + " | team " + " ".join(f"{o[3]:.3f}" for o in out), flush=True)
    return m[0]
def run_all(fn, names=NAMES, jobs=4):
    with Pool(jobs) as p: return dict(zip(names, p.map(fn, names)))
