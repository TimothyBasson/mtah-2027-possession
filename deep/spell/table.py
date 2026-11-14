"""logs/heads.log -> one line per variant: held-out spellF1 by provider, seeds averaged."""
import collections
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(os.path.join(ROOT, "logs", "heads.log")):
    w = line.split()
    rows[w[1]][w[4]].append([float(w[7]), float(w[9]), float(w[12]), float(w[14])])
print(f"{'variant':34s} seeds  raw: idsse metrica skillc | mean    control: same           | mean   val raw/control")
for name, r in rows.items():
    if len(r) < 3 or (len(sys.argv) > 1 and not any(a in name for a in sys.argv[1:])):
        continue
    m = np.array([np.mean(r[f], 0) for f in ("idsse", "metrica", "skillcorner")])
    n = min(len(v) for v in r.values())
    print(f"{name:34s} {n:3d}    " + " ".join(f"{x:.3f}" for x in m[:, 0]) + f" | {m[:, 0].mean():.3f}   "
          + " ".join(f"{x:.3f}" for x in m[:, 1]) + f" | {m[:, 1].mean():.3f}   {m[:, 2].mean():.3f} {m[:, 3].mean():.3f}")
