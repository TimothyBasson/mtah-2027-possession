"""Label BAR-ATH with the learned model and write the submission.

    python deep/run.py [--tag v2]      # -> evaluation/results/deep.csv

Uses the network trained on all 30 outside matches (deep/train.py --fold none). Players only.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), ROOT]
import data as D  # noqa: E402
import decode  # noqa: E402
from net import PossessionNet  # noqa: E402
from train import predict_half  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)  # what two of the three held-out folds picked


def probabilities(tag):
    import load
    players, ball, team, halves = load.load()
    net = PossessionNet(nfeat=D.NFEAT)
    net.load_state_dict(torch.load(os.path.join(HERE, "models", tag + ".pt")))  # train.py --fold none
    probs = []
    for a, b in halves:
        X, M, bxy, fr = D.half_tensors(players, ball, team, a, b, load.PITCH, load.FPS)
        probs.append(predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy)))
    return probs, halves, len(players)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", default="deep.csv")
    a = ap.parse_args()
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    probs, halves, n = probabilities(a.tag)
    os.makedirs(os.path.join(ROOT, "data", "cache", "deep", a.tag), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "data", "cache", "deep", a.tag, "bar_ath_probs.npz"),
                        **{f"p{i}": p for i, p in enumerate(probs)})
    pred = decode.labels_from(probs, halves, n, **DECODE)
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):  # graded frames before a kickoff are D
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
