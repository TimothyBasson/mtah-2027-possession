"""Blend two runs' held-out probabilities: out = (1-a) * A + a * B, written as a new run tag.

    python alldata/briefsafe/blend.py <tagA> <tagB> <a> <out> [--geo]
"""
import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "data", "cache", "alldata", "runs")
A, B, a, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
geo = "--geo" in sys.argv
for f in glob.glob(os.path.join(RUNS, A, "*", "*.npz")):
    fold, name = f.split(os.sep)[-2:]
    g = os.path.join(RUNS, B, fold, name)
    if not os.path.exists(g):
        continue
    za, zb = np.load(f), np.load(g)
    o = {}
    for k in za.files:
        if k == "kind":
            o[k] = za[k]; continue
        p, q = za[k], zb[k]
        if geo:
            r = np.exp((1 - a) * np.log(np.clip(p, 1e-6, 1)) + a * np.log(np.clip(q, 1e-6, 1)))
            o[k] = r / r.sum(-1, keepdims=True)
        else:
            o[k] = (1 - a) * p + a * q
    os.makedirs(os.path.join(RUNS, out, fold), exist_ok=True)
    np.savez_compressed(os.path.join(RUNS, out, fold, name), **o)
