"""Control: sharpen saved probabilities (p ** (1 / temp), renormalised) into a new tag, to tell
entropy minimisation's gain apart from what plain sharpening does to the fixed decode.

    python alldata/selftrain/sharpen.py pw1e pw1e_t05 0.5
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "data", "cache", "alldata", "runs")
src, out, temp = sys.argv[1], sys.argv[2], float(sys.argv[3])
for f in glob.glob(os.path.join(RUNS, src, "*", "*.npz")):
    z = np.load(f)
    o = {}
    for k in z.files:
        if k == "kind":
            continue
        p = np.clip(z[k], 1e-6, 1) ** (1 / temp)
        o[k] = p / p.sum(-1, keepdims=True)
    dst = os.path.join(RUNS, out, os.path.relpath(f, os.path.join(RUNS, src)))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, kind=z["kind"], **o)
