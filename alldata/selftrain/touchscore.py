"""Held-out spellF1 of saved runs decoded with the submission's touch evidence (alldata/touch/run.py:
detector t1, threshold 0.3, weight 1, hold 3 s), per provider, labels as built / folded.

    python alldata/selftrain/touchscore.py pw1e tf2 ...
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "alldata", "touch"))
import common as C  # noqa: E402
import tuse  # noqa: E402


def one(args):
    tags, fold, name = args
    raw, fo, spans = C.truths(name)
    n = len(raw)
    q = tuse.touch_probs("t1", fold, name, 1, n, spans)
    f, tm, sc = tuse.events(q, 0.3)
    ev, _ = tuse.evidence(n, f, tm, sc, 1.0, 3.0)
    out = []
    for tag in tags:
        z = np.load(os.path.join(C.RUNS, tag, fold, name + ".npz"))
        pr = [z[f"s{i}"] for i in range(len(spans))]
        pf = [z[f"fs{i}"] for i in range(len(spans))]
        r = C.evaluate(name, C.decode_ev(pr, spans, n, ev, **C.RAW), C.decode_ev(pf, spans, n, ev, **C.FOLDED))
        out.append(r)
    return name, out


if __name__ == "__main__":
    tags = sys.argv[1:]
    ms = [(fo, os.path.basename(p)[:-4]) for fo in C.PROVIDERS
          for p in sorted(__import__("glob").glob(os.path.join(C.RUNS, tags[0], fo, "*.npz")))
          if str(np.load(p)["kind"]) == "test"]
    res = dict(C.pool_map(one, [(tags, fo, n) for fo, n in ms], 4))
    for i, t in enumerate(tags):
        C.report(t, {n: r[i] for n, r in res.items()}, {n: r[0] for n, r in res.items()} if i else None)
