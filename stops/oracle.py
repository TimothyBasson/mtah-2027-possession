"""How much spellF1 sits beside stoppages: the model's held-out probabilities with pieces of the
truth written in, then the shipped decode.

    python stops/oracle.py bt [bts1 ...]
"""

import sys
from multiprocessing import Pool

import numpy as np

import common as C
from common import decode

IDX = {"H": 0, "A": 1, "D": 2}


def runs(lab):
    e = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    s = np.r_[0, e]
    return [(a, b, lab[a]) for a, b in zip(s, np.r_[e, len(lab)])]


def near_edges(truth, w, kind):
    """Frames within w frames of a boundary of the kind: 'stop' live->D, 'start' D->live."""
    m = np.zeros(len(truth), bool)
    for a, b, k in runs(truth):
        if k != "D":
            continue
        if kind in ("stop", "both") and a > 0 and truth[a - 1] in "HA":
            m[max(0, a - w):a + w] = True
        if kind in ("start", "both") and b < len(truth) and truth[b] in "HA":
            m[max(0, b - w):b + w] = True
    return m


def write(P, truth, where, what):
    """P (n, 3) full-rate probabilities; on frames `where`, write truth in: 'dead' = the alive/dead
    split only, 'all' = the label itself."""
    P = P.copy()
    d = where & (truth == "D")
    live = where & ((truth == "H") | (truth == "A"))
    P[d] = [1e-6, 1e-6, 1]
    if what == "dead":
        P[live, 2] = 1e-6
    else:
        for c in "HA":
            P[where & (truth == c)] = np.eye(3)[IDX[c]] + 1e-6
    return P


def variants(truth, P):
    n = len(truth)
    yes = np.ones(n, bool)
    isd = truth == "D"
    out = {"model": P,
           "true dead/alive": write(P, truth, yes, "dead"),
           "no missed stops": write(P, truth, isd, "dead"),
           "no false stops": write(P, truth, ~isd, "dead")}
    for w in (1, 3):
        for kind in ("stop", "start", "both"):
            out[f"d/a {kind} ±{w}s"] = write(P, truth, near_edges(truth, w * 25, kind), "dead")
    for w in (3,):
        for kind in ("stop", "start", "both"):
            out[f"all {kind} ±{w}s"] = write(P, truth, near_edges(truth, w * 25, kind), "all")
    return out


def one(args):
    name, tags, T = args
    spans = T["spans"]
    res = []
    for tag in tags:
        pr = C.probs(tag, name)
        full = np.full((len(T["raw"]), 3), 1 / 3)
        for p, (a, b) in zip(pr, spans):
            full[a:b + 1] = decode.upsample(p, b - a + 1)
        row = []
        for view in ("raw", "fold"):
            truth = T[view]
            V = variants(truth, full)
            keys = list(V)
            for k in keys:
                pred = np.full(len(truth), "X", "<U1")
                for a, b in spans:
                    lp = np.log(np.clip(V[k][a:b + 1], 1e-6, 1))
                    path = decode.absorb(decode.viterbi(lp, C.DECODE["cost"]), {0: 25, 1: 25, 2: 25})
                    pred[a:b + 1] = decode.LETTERS[path]
                row.append(C.spell(truth, pred))
        res.append(row)
    return name, np.mean(res, 0), keys


if __name__ == "__main__":
    tags = sys.argv[1:] or ["bt"]
    T = C.truths()
    with Pool(4) as pool:
        out = pool.map(one, [(n, tags, T[n]) for n in sorted(T)])
    names = [n for n, _, _ in out]
    keys = out[0][2]
    R = np.stack([r for _, r, _ in out]).reshape(len(names), 2, len(keys))
    for vi, view in enumerate(("labels as given", "folded labels")):
        print(f"\n{view}  ({'+'.join(tags)})")
        print(f"{'':22s}   idsse  metrica  skillc    mean   gain")
        base = None
        for ki, k in enumerate(keys):
            pm_ = [R[[n.startswith(p) for n in names], vi, ki].mean() for p in C.PROVIDERS]
            m = np.mean(pm_)
            base = m if base is None else base
            print(f"{k:22s}  " + "  ".join(f"{x:.3f}" for x in pm_) + f"   {m:.3f}  {m - base:+.3f}")
