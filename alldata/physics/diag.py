"""Where a run loses spellF1: its score given the true stoppages, and given the true team.

    python alldata/physics/diag.py --run s3h

"true stoppages": dead exactly where the labels are dead, the run's own team call elsewhere.
"true team": the run's own stoppages, the true team wherever both call the ball alive.
Also splits true possessions by length and says how many are matched at IoU 0.5.
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
from score import truth  # noqa: E402
from train import pred_dir  # noqa: E402
import possession_metrics as pm  # noqa: E402

DEC = dict(cost=4.0, min_team=1.0, min_dead=1.0)
BINS = (0, 2, 5, 10, 1e9)


def one(args):
    name, run = args
    lab, fol, spans = truth(name)
    z = np.load(os.path.join(pred_dir(run), name + ".npz"))
    probs = [z[f"p{i}"] for i in range(len(spans))]
    pred = DC.labels_from(probs, spans, len(lab), **DEC)
    # team-only decode: the dead column removed
    team_only = np.full(len(lab), "X", "<U1")
    for p, (a, b) in zip(probs, spans):
        q = p.copy()
        q[:, 2] = 1e-9
        team_only[a:b + 1] = DC.LETTERS[DC.decode_half(q, b - a + 1, **DEC)]
    od = np.where(lab == "D", "D", np.where(lab == "X", pred, team_only))
    both = np.isin(lab, ("H", "A")) & np.isin(pred, ("H", "A"))
    ot = np.where(both, lab, pred)
    t = pm.label_codes(lab)
    sc = [pm.evaluate(t, pm.label_codes(x), bootstrap_n=0)["spell_f1_mean"] for x in (pred, od, ot)]
    # matched share of true possessions by length at IoU 0.5
    runs = pm.valid_runs(t)
    filled = pm.fill_abstentions(pm.label_codes(pred), runs)
    gs, ps = pm.possession_spells(t, runs), pm.possession_spells(filled, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
    ln = np.array([(s[1] - s[0]) / 25 for s in gs]) if len(gs) and not hasattr(gs, "dtype") else None
    by = []
    if ln is None:
        g = np.asarray(gs)
        ln = (g[:, 1] - g[:, 0]) / 25
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ln >= lo) & (ln < hi)
        by += [m.sum(), np.asarray(flags)[m].sum()]
    return name, sc, by, len(ps) / max(len(gs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    names = [n for n in C.names() if os.path.exists(os.path.join(pred_dir(a.run), n + ".npz"))]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, [(n, a.run) for n in names]))
    print(f"{'':12s} as is   true stoppages   true team   matched at IoU 0.5 by true length: <2 s, 2-5, 5-10, 10+")
    for p in C.PROVIDERS:
        r = [x for x in res if C.provider(x[0]) == p]
        sc = np.mean([x[1] for x in r], 0)
        by = np.sum([x[2] for x in r], 0)
        print(f"{p:12s} {sc[0]:.3f}   {sc[1]:.3f}            {sc[2]:.3f}       "
              + "  ".join(f"{by[2 * i + 1] / max(by[2 * i], 1):.2f} of {by[2 * i] / len(r):.0f}" for i in range(4)))


if __name__ == "__main__":
    main()
