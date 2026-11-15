"""Predict a tag's held-out and validation matches again from its saved fold nets.

    python deep/predict.py --tag v1 --out v1o            # after a change to deep/data.py
    python deep/predict.py --tag v1 --out v1k --knock    # players lost as BAR-ATH's tracker loses them

Reads data/cache/deep/<tag>/<fold>/net.pt and the list of matches saved beside it, and writes
data/cache/deep/<out>/<fold>/<match>.npz in the layout deep/train.py writes, for deep/decode.py.
--knock removes players with external/score.py drop_players (about 7% of player-frames, in runs
and bursts, speeds recomputed) before the tensors are built. Nothing is trained.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "external"), os.path.join(ROOT, "possession")]
import data as D  # noqa: E402
from net import PossessionNet  # noqa: E402
from train import predict_half  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--knock", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    base = os.path.join(ROOT, "data", "cache", "deep")
    for fold in sorted(os.listdir(os.path.join(base, a.tag))):
        src = os.path.join(base, a.tag, fold)
        if not os.path.exists(os.path.join(src, "net.pt")):
            continue
        net = PossessionNet(nfeat=D.NFEAT)
        net.load_state_dict(torch.load(os.path.join(src, "net.pt")))
        os.makedirs(os.path.join(base, a.out, fold), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
            name = os.path.basename(f)
            d = np.load(D.match_path(name[:-4]))
            halves = [tuple(int(v) for v in h) for h in d["halves"]]
            players = d["players"]
            if a.knock:
                from score import drop_players
                players = drop_players(players, halves, int(d["fps"]), name)
            probs = {}
            for i, (s, e) in enumerate(halves):
                X, M, bxy, _ = D.half_tensors(players, d["ball"], d["team"], s, e, tuple(d["pitch"]), int(d["fps"]))
                probs[f"p{i}"] = predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy))
            np.savez_compressed(os.path.join(base, a.out, fold, name), kind=str(np.load(f)["kind"]), **probs)
            print(fold, name, flush=True)


if __name__ == "__main__":
    main()
