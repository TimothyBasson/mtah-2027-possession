"""A run's fold nets on the current feature cache, saved under a new tag: what a model trained on
one kind of ball does when shown another.

    python alldata/learned/repredict.py c1 c1_barlike
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats  # noqa: E402
import fit as T  # noqa: E402

src, dst = sys.argv[1:3]
torch.set_num_threads(1)
ms = T.load_matches()
for fold in ("idsse", "metrica", "skillcorner"):
    ck = torch.load(os.path.join(T.CACHE, "runs", src, fold, "net.pt"), map_location="cpu")
    cols = [ms[0]["names"].index(n) for n in ck["names"]]
    net = T.Net(len(cols), ck["d"])
    net.load_state_dict(ck["sd"])
    sym = feats.symmetry(ck["names"])
    os.makedirs(os.path.join(T.CACHE, "runs", dst, fold), exist_ok=True)
    for m in T.split(ms, fold)[2]:
        o = {}
        for i, h in enumerate(m["halves"]):
            for k, v in (("p", 0), ("s", 1)):
                q = T.predict(net, h["F"][v][:, cols], sym)
                o[f"{k}{i}"], o[f"f{k}{i}"] = q[:, :3], q[:, 3:]
        np.savez_compressed(os.path.join(T.CACHE, "runs", dst, fold, m["name"] + ".npz"), kind="test", **o)
