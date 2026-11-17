"""One model's team split on another set of models' dead-ball probability, held out.
    python stops/mixdead.py bt  bt,bts1,btk,btc,v2,v2s1
"""
import sys, os
from multiprocessing import Pool
import numpy as np
import common as C
from common import decode


def one(args):
    name, base, sets, T = args
    row = []
    pb = C.probs(base, name)
    for tags in sets:
        out = []
        for i, p in enumerate(pb):
            if tags:
                pd = np.mean([C.probs(t, name)[i][:, 2] for t in tags], 0)
                team = p[:, :2] / np.clip(p[:, :2].sum(1, keepdims=True), 1e-6, None)
                p = np.c_[team * (1 - pd)[:, None], pd]
            out.append(p)
        pred = decode.labels_from(out, T["spans"], len(T["raw"]), **C.DECODE)
        row += [C.spell(T["raw"], pred), C.spell(T["fold"], pred)]
    return name, row


if __name__ == "__main__":
    base = sys.argv[1]
    sets = [None] + [s.split(",") for s in sys.argv[2:]]
    T = C.truths()
    with Pool(int(os.environ.get("POOL", 4))) as pool:
        out = pool.map(one, [(n, base, sets, T[n]) for n in sorted(T)])
    cols = []
    for s in [base] + sys.argv[2:]:
        cols += [s[:9] + " raw", s[:9] + " fold"]
    C.table([r for _, r in out], [n for n, _ in out], cols)
