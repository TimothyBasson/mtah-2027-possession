"""Tim's HMM v8 (tim.py's held-out labels) under this folder's yardstick: spellF1 per provider on
the labels as built and folded, ball as recorded.    python alldata/learned/bench.py"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
import possession_metrics as pm  # noqa: E402

rows = {}
for f in sorted(glob.glob(os.path.join(S.ROOT, "data", "cache", "alldata", "tim", "*_*.npz"))):
    name = os.path.basename(f)[:-4]
    raw, fold, _ = S.truths(name)
    pred = pm.label_codes(np.load(f)["labels"])
    rows[name] = [pm.evaluate(t, pred, bootstrap_n=0)["spell_f1_mean"] for t in (raw, fold)]
for k, title in ((0, "labels as built"), (1, "folded labels")):
    per = [np.mean([v[k] for n, v in rows.items() if n.startswith(p)]) for p in ("idsse", "metrica", "skillcorner")]
    print(f"hmm v8, {title}: idsse {per[0]:.3f}  metrica {per[1]:.3f}  skillcorner {per[2]:.3f}  mean {np.mean(per):.3f}")
