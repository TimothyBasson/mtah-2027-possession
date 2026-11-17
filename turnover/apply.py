"""The detector's turnovers as evidence in the decode of the base model, held out by provider.

    python turnover/apply.py --det t1 [--base bt,bts1]

At a frame where the detector says "the ball passes to home", switching from away to home costs
less by w * (its log-odds above a floor); elsewhere a direct H<->A switch costs ha_cost. The
settings are picked on each fold's validation matches (training providers), then scored on the
held-out provider, on the labels as given and folded to the control rule.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from common import sh  # noqa: E402

GRID = [None] + [dict(key=k, w=w, ha=ha, floor=fl, gate=gate) for k in ("ev", "evf") for w in (2, 4, 8) for ha in (4, 8) for fl in (0.02, 0.05) for gate in (0, 1)]


def evidence(z, i, key, n, w, floor):
    p = z[f"{key}{i}"].astype(np.float64)
    t = np.arange(len(p)) * 2
    q = np.stack([np.interp(np.arange(n), t, p[:, k]) for k in (1, 2)], 1)
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return w * np.clip(np.log(q / (1 - q)) - np.log(floor / (1 - floor)), 0, None)


def decode(probs, spans, n, z, g):
    out = np.full(n, "X", "<U1")
    for i, (p, (a, b)) in enumerate(zip(probs, spans)):
        lp = np.log(np.clip(C.decode.upsample(p, b - a + 1), 1e-6, 1))
        if g is None:
            path = C.viterbi_ev(lp, C.COST)
        else:
            e = evidence(z, i, g["key"], b - a + 1, g["w"], g["floor"])
            if g.get("gate"):  # no turnover where the base model sees a stoppage within a second
                dead = np.exp(lp[:, 2]) > 0.3
                near = np.convolve(dead, np.ones(51), "same") > 0
                e[near] = 0
            path = C.viterbi_ev(lp, C.COST, e, g["ha"])
        path = C.absorb(path, C.MIN_TEAM, C.MIN_DEAD)
        out[a:b + 1] = C.LETTERS[path]
    return out


def one(args):
    fold, name, base, det = args
    d = sh.load(os.path.join(C.ROOT, "data", "external", name + ".npz"))
    spans = [tuple(h) for h in d["halves"]]
    n = len(d["labels"])
    ps = [np.load(os.path.join(C.ROOT, "data", "cache", "deep", t, fold, name + ".npz")) for t in base]
    probs = [np.mean([p[f"p{i}"] for p in ps], 0) for i in range(len(spans))]
    z = np.load(os.path.join(C.ROOT, "data", "cache", "turnover", det, fold, name + ".npz"))
    truths = (d["labels"], C.control_labels(d))
    rows = []
    for g in GRID:
        pred = decode(probs, spans, n, z, g)
        pred[d["labels"] == "X"] = "X"
        rows.append([sh.score(gt, pred)[0] for gt in truths])
    return fold, name, str(z["kind"]), np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="t1")
    ap.add_argument("--base", default="bt,bts1")
    ap.add_argument("--folds", default="idsse,metrica,skillcorner")
    a = ap.parse_args()
    base = a.base.split(",")
    jobs = []
    for fold in a.folds.split(","):
        for f in sorted(os.listdir(os.path.join(C.ROOT, "data", "cache", "turnover", a.det, fold))):
            if f.endswith(".npz") and all(os.path.exists(os.path.join(C.ROOT, "data", "cache", "deep", t, fold, f)) for t in base):
                jobs.append((fold, f[:-4], base, a.det))
    with Pool(4) as pool:
        res = pool.map(one, jobs)
    np.save(os.path.join(C.ROOT, "data", "cache", "turnover", a.det, "apply_" + "_".join(base) + ".npy"),
            np.array([(f, n, k, r) for f, n, k, r in res], dtype=object), allow_pickle=True)
    tot = {}
    for fold in a.folds.split(","):
        val = [(n, r) for f, n, k, r in res if f == fold and k == "val"]
        test = np.array([r for f, n, k, r in res if f == fold and k == "test"])
        provs = sorted({n.split("_")[0] for n, _ in val})
        v = np.mean([np.mean([r for n, r in val if n.startswith(p)], 0) for p in provs], 0)   # (grid, truths)
        t = test.mean(0)
        pick = int(v.sum(1).argmax())
        tot[fold] = (t, pick, v)
        print(f"\n{fold}: {len(test)} held-out matches. base raw {t[0, 0]:.3f} folded {t[0, 1]:.3f} | validation picks {GRID[pick]}: "
              f"raw {t[pick, 0]:.3f} folded {t[pick, 1]:.3f} | validation gain {v[pick] - v[0]}")
    T = np.mean([t for t, _, _ in tot.values()], 0)
    print("\nOne setting everywhere, provider mean (raw, folded) and per-provider change, sorted by mean gain:")
    order = np.argsort(-(T - T[0]).sum(1))
    for gi in list(order[:8]) + [0]:
        ch = "  ".join(f"{f[:3]} {tot[f][0][gi, 0] - tot[f][0][0, 0]:+.3f}/{tot[f][0][gi, 1] - tot[f][0][0, 1]:+.3f}" for f in tot)
        print(f"  {T[gi, 0]:.3f} {T[gi, 1]:.3f}   {ch}   {GRID[gi]}")
    V = np.mean([v for _, _, v in tot.values()], 0)
    gi = int(V.sum(1).argmax())
    print("one setting, picked on the folds' validation matches: %s raw %.3f folded %.3f; per provider " % (GRID[gi], T[gi, 0], T[gi, 1])
          + "  ".join(f"{f[:3]} {tot[f][0][gi, 0] - tot[f][0][0, 0]:+.3f}/{tot[f][0][gi, 1] - tot[f][0][0, 1]:+.3f}" for f in tot))
    print("picked per fold on validation: raw %.3f folded %.3f (base %.3f %.3f)" % (
        np.mean([t[p, 0] for t, p, _ in tot.values()]), np.mean([t[p, 1] for t, p, _ in tot.values()]), T[0, 0], T[0, 1]))


if __name__ == "__main__":
    main()
