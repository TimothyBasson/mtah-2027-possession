"""What filling gaps changes on BAR-ATH itself, for any model in deep/models/.

    python adapt/bar_ath.py [--tag v2] [--fill 15]

Labels BAR-ATH twice with the same network and decode, once on the tracking as recorded and once
with gaps up to --fill seconds joined (adapt/impute.py), and prints how the two differ by how many
players the tracker had. The public 5 minutes are scored as a check only: they have no frame with
fewer than 20 tracked, so they cannot show what the filling is for.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "deep"), os.path.join(ROOT, "possession"),
                os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation"), ROOT]
import data as D  # noqa: E402
import decode  # noqa: E402
import impute  # noqa: E402
import load  # noqa: E402
import possession_metrics as pm  # noqa: E402
from net import PossessionNet  # noqa: E402
from train import predict_half  # noqa: E402


def label(net, players, team, halves, n, dec):
    probs = []
    for a, b in halves:
        X, M, bxy, _ = D.half_tensors(players, np.full((n, 3), np.nan, np.float32), team, a, b, load.PITCH, load.FPS)
        probs.append(predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy)))
    return decode.labels_from(probs, halves, n, **dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--fill", type=float, default=15)
    a = ap.parse_args()
    torch.set_num_threads(4)
    players, _, team, halves = load.load()  # the ball is not passed on
    net = PossessionNet(nfeat=D.NFEAT)
    net.load_state_dict(torch.load(os.path.join(ROOT, "deep", "models", a.tag + ".pt")))
    n = len(players)
    dec = dict(cost=4, min_team=1, min_dead=1)
    base = label(net, players, team, halves, n, dec)
    p2, t2 = impute.fill(players, team, halves, load.FPS, a.fill)
    filled = label(net, p2, t2, halves, n, dec)
    tracked = np.isfinite(players[..., 0]).sum(1)
    g = base != "X"
    print(f"agree on {np.mean(base[g] == filled[g]):.3f} of graded frames")
    for lo, hi in ((0, 14), (15, 19), (20, 21), (22, 99)):
        s = g & (tracked >= lo) & (tracked <= hi)
        print(f"{lo:2d}-{hi:2d} tracked: {s.sum() / g.sum():.3f} of frames, called dead {np.mean(base[s] == 'D'):.2f} "
              f"-> {np.mean(filled[s] == 'D'):.2f}, labels differ on {np.mean(base[s] != filled[s]):.3f}")
    import pandas as pd
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    truth = np.full(n, "X", "<U1")
    truth[gt["frame"].values] = gt["label"].values
    for name, pred in (("as recorded", base), (f"gaps to {a.fill:g} s filled", filled)):
        r = pm.evaluate(pm.label_codes(truth), pm.label_codes(np.where(truth == "X", "X", pred)), bootstrap_n=0)
        print(f"public 5 minutes, check only, {name}: spellF1 {r['spell_f1_mean']:.3f} macroF1 {r['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
