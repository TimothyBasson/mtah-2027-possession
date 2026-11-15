"""Held-out spellF1 of saved probabilities, on the labels as built and folded.

    python alldata/physics/score.py --runs base [other ...] [--grid]

Each provider's mean, and the mean of the three (each provider counts once). With --grid the
decode settings are swept; the fair figure picks a provider's setting on the other two providers.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import decode as DC  # noqa: E402
from build import table_path  # noqa: E402
from train import pred_dir  # noqa: E402
import possession_metrics as pm  # noqa: E402

DEFAULT = [dict(cost=4.0, min_team=1.0, min_dead=1.0)]
GRID = [dict(cost=c, cost_dead=cd, min_team=t, min_dead=d)
        for c in (2.0, 4.0, 8.0) for cd in (None, 2 * c) for t, d in ((0.5, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0))]


def truth(name, mode="barath", tag="f4"):
    z = np.load(table_path(name, mode, tag))
    return z["labels"], z["folded"], [tuple(int(v) for v in h) for h in z["halves"]]


def one(args):
    name, runs, combos = args
    lab, fol, spans = truth(name)
    tr, tf = pm.label_codes(lab), pm.label_codes(fol)
    out = []
    for run in runs:
        z = np.load(os.path.join(pred_dir(run), name + ".npz"))
        probs = [z[f"p{i}"] for i in range(len(spans))]
        row = []
        for c in combos:
            pred = pm.label_codes(DC.labels_from(probs, spans, len(lab), **c))
            r, f = pm.evaluate(tr, pred, bootstrap_n=0), pm.evaluate(tf, pred, bootstrap_n=0)
            row.append((r["spell_f1_mean"], f["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"],
                        r["team_acc_both_alive"], r["n_pred_spells"] / max(r["n_gt_spells"], 1)))
        out.append(row)
    return name, out


def table(runs, combos, workers=4):
    names = [n for n in C.names() if all(os.path.exists(os.path.join(pred_dir(r), n + ".npz")) for r in runs)]
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(one, [(n, runs, combos) for n in names]))
    return names, np.array([res[n] for n in names])  # match, run, combo, metric


def by_provider(names, R):
    return np.stack([R[[i for i, n in enumerate(names) if C.provider(n) == p]].mean(0) for p in C.PROVIDERS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    combos = GRID if a.grid else DEFAULT
    names, R = table(a.runs, combos)
    P = by_provider(names, R)  # provider, run, combo, metric
    cols = ("raw", "folded", "macroF1", "alive/dead", "team", "n/true")
    for ri, run in enumerate(a.runs):
        if a.grid:
            print(f"== {run}: decode grid, mean of providers (raw / folded)")
            for ci, c in enumerate(combos):
                print(f"  {str(c):78s} {P[:, ri, ci, 0].mean():.3f} / {P[:, ri, ci, 1].mean():.3f}   "
                      + " ".join(f"{P[pi, ri, ci, 0]:.3f}" for pi in range(3)))
            for k, nm in ((0, "raw"), (1, "folded")):
                fair = []
                for pi in range(3):
                    others = [q for q in range(3) if q != pi]
                    pick = int(P[others, ri, :, k].mean(0).argmax())
                    fair.append(P[pi, ri, pick, k])
                print(f"  {nm}: setting picked on the other two providers: "
                      + " / ".join(f"{x:.3f}" for x in fair) + f"   mean {np.mean(fair):.3f}")
        else:
            print(f"== {run}")
            print(f"  {'':12s}" + "".join(f"{c:>11s}" for c in cols))
            for pi, p in enumerate(C.PROVIDERS):
                print(f"  {p:12s}" + "".join(f"{x:11.3f}" for x in P[pi, ri, 0]))
            print(f"  {'mean':12s}" + "".join(f"{x:11.3f}" for x in P[:, ri, 0].mean(0)))
            if ri:
                for k, nm in ((0, "raw"), (1, "folded")):
                    dl = R[:, ri, 0, k] - R[:, 0, 0, k]
                    print(f"  {nm} vs {a.runs[0]} per match: {dl.mean():+.3f} (se {dl.std(ddof=1) / np.sqrt(len(dl)):.3f}), "
                          f"up on {(dl > 0).sum()} of {len(dl)}")


if __name__ == "__main__":
    main()
