"""Train the segment corrector on all outside matches and save it for BAR-ATH.

    python alldata/corrector/train.py [--name segmix] [--target mix] [--sc-share 0] [--seeds 0 1]

Candidates and rows come from segs.py's cache in the shipped configuration (the two heads
averaged, decode cost 4, 1 s / 1 s). Writes alldata/corrector/models/<name>_s<seed>.txt.
"""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import segfit as SF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="segmix")
    ap.add_argument("--segs", default="segs_pw1e_v1")
    ap.add_argument("--target", default="mix", choices=["", "f", "mix", "team"])
    ap.add_argument("--sc-share", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    z = np.load(os.path.join(SF.CACHE, "corrector", a.segs + ".npz"))
    cols = [str(c) for c in z["names"]]
    names = sorted({k.split("|")[0] for k in z.files if "|" in k})
    if a.sc_share <= 0:
        names = [n for n in names if not n.startswith("skillcorner")]
    G = {n: dict(to=z[f"{n}|ato"], rows=z[f"{n}|arows"], y=z[f"{n}|ay"], u=z[f"{n}|au"], uf=z[f"{n}|auf"]) for n in names}
    X = np.vstack([G[n]["rows"] for n in names])
    y = np.concatenate([SF.target(G[n], a.target) for n in names])
    w = SF.weights(names, "none", a.sc_share, G)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    for seed in a.seeds:
        m = lgb.train(dict(SF.PARAMS, seed=seed, deterministic=True, force_row_wise=True, num_threads=1), lgb.Dataset(X, y, weight=w, feature_name=cols), a.rounds)
        f = os.path.join(HERE, "models", f"{a.name}_s{seed}.txt")
        m.save_model(f)
        print("wrote", f, "from", len(y), "candidates,", int(y.sum()), "positive")


if __name__ == "__main__":
    main()
