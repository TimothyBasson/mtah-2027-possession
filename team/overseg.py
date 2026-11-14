"""Ceiling of naming finely cut stage-1 pieces with the true team, then merging (folded labels)."""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import decode
from oracle import maj

CUTS = [dict(cost=4, min_team=2, min_dead=3), dict(cost=4, min_team=1, min_dead=3), dict(cost=1, min_team=1, min_dead=3),
        dict(cost=1, min_team=0.4, min_dead=3), dict(cost=0.25, min_team=0.4, min_dead=3), dict(cost=0, min_team=0.2, min_dead=3)]
TAGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["v2"]

def one(name):
    raw, fold, spans = C.truth(name)
    ps = [C.stage1(t, name) for t in TAGS]
    probs = [np.mean([p[i] for p in ps], 0) for i in range(len(spans))]
    row = []
    for cut in CUTS:
        pred = decode.labels_from(probs, spans, len(raw), **cut)
        b = pred.copy(); n = 0
        for s, e, v in C.runs(pred):
            if v in ("H", "A"):
                b[s:e] = maj(fold, s, e, v); n += 1
        # the 2 s minimum again after naming
        code = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[c] for c in b], np.int8)
        for a, bb in spans:
            code[a:bb + 1] = decode.absorb(code[a:bb + 1], {0: 50, 1: 50, 2: 75, 3: 0})
        b2 = np.array(list("HADX"))[code]
        row.append((C.spell_f1(fold, pred), C.spell_f1(fold, b), C.spell_f1(fold, b2), n))
    return row

if __name__ == "__main__":
    names = [os.path.basename(p)[:-4] for p in C.paths()]
    with Pool(4) as pool:
        R = np.array(pool.map(one, names))
    for i, cut in enumerate(CUTS):
        print(cut)
        for j, k in enumerate(("as decoded", "pieces true-named", "true-named + 2 s min", "pieces/match")):
            print(f"   {k:22s} " + "  ".join(f"{x:.3f}" if j < 3 else f"{x:.0f}" for x in C.provider_mean(names, R[:, i, j])))
