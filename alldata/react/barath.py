"""The reaction correction on BAR-ATH: the public 5 minutes (a check: 17 possessions) and the
possessions it draws per half. Writes nothing into evaluation/results.

    python alldata/learned/run.py            # first: the submission's probabilities on BAR-ATH
    python alldata/react/barath.py [--weight-brief 5] [--scales 0 0.5 1]

The correction is fitted on IDSSE's and Metrica's held-out matches (resid.py --no-sc) and applied
to the submission's probabilities (tpsN + tpsNs1). Scale 0 is the submission itself.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rdata  # noqa: E402
import react as R  # noqa: E402
import resid  # noqa: E402
from common import pm, decode  # noqa: E402

sys.path += [os.path.join(ROOT, "possession"), ROOT]


class Bar:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-brief", type=float, default=5.0)
    ap.add_argument("--scales", type=float, nargs="+", default=[0, 0.5, 1.0])
    ap.add_argument("--with-base", nargs="?", const=True, default=False)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    import load
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    players, ball, team, halves = load.load()
    z = np.load(os.path.join(ROOT, "data", "cache", "alldata", "bar_ath", "tpsN+tpsNs1.npz"))
    m = Bar()
    m.probs = [z[f"p{i}"] for i in range(len(halves))]
    m.R = [R.half_rows(players, ball[:, :2], team, a0, b0, load.PITCH, load.FPS)[0] for a0, b0 in halves]
    ms = [x for x in rdata.load_all() if x.fold != "skillcorner"]
    bo = [resid.fit(ms, t, s, 100, a.weight_brief, a.with_base, 0.05) for s in a.seeds for t in ("built", "folded")]
    gt = pm.label_codes(pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")).label.values)
    for sc in a.scales:
        probs = resid.apply(bo, m, a.with_base, sc)
        pred = decode.labels_from(probs, halves, len(players), **rdata.DECODE)
        for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
            pred[s:kick] = "D"
        r = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=0)
        counts = []
        for a0, b0 in halves:
            x = pred[a0:b0 + 1]
            e = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
            counts.append(int(np.isin(x[e], ["H", "A"]).sum()))
        ov = pm.spell_overlaps(pm.possession_spells(gt, pm.valid_runs(gt)), pm.possession_spells(pm.label_codes(pred), pm.valid_runs(gt)))
        ng = len(pm.possession_spells(gt, pm.valid_runs(gt)))
        print(f"scale {sc:g}: public window spellF1 {r['spell_f1_mean']:.3f}, found at IoU 0.5 {int(pm.matched_gt_flags(ov, ng, 0.5).sum())} of {ng}; possessions per half {counts}")


if __name__ == "__main__":
    main()
