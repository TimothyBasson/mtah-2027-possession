"""Average the saved probabilities of several runs into a new tag.

    python alldata/learned/ens.py c1e c1 c1s1
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "data", "cache", "alldata", "runs")
out, tags = sys.argv[1], sys.argv[2:]
for f in glob.glob(os.path.join(RUNS, tags[0], "*", "*.npz")):
    rel = os.path.relpath(f, os.path.join(RUNS, tags[0]))
    fs = [os.path.join(RUNS, t, rel) for t in tags]
    if not all(os.path.exists(x) for x in fs):
        continue
    zs = [np.load(x) for x in fs]
    o = {k: np.mean([z[k] for z in zs], 0) for k in zs[0].files if k != "kind"}
    os.makedirs(os.path.dirname(os.path.join(RUNS, out, rel)), exist_ok=True)
    np.savez_compressed(os.path.join(RUNS, out, rel), kind=zs[0]["kind"], **o)
