"""Where the turnovers go missing: true (folded) possessions by length, what stage 1 says inside them,
and what finding only the longer ones would be worth."""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import decode
TAGS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["v2"]
BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9))

def one(name):
    raw, fold, spans = C.truth(name)
    ps = [C.stage1(t, name) for t in TAGS]
    probs = [np.mean([p[i] for p in ps], 0) for i in range(len(spans))]
    pred = decode.labels_from(probs, spans, len(raw), **C.DECODE)
    p25 = np.full((len(raw), 3), 1 / 3)
    for p, (a, b) in zip(probs, spans):
        p25[a:b + 1] = decode.upsample(p, b - a + 1)
    rows = []
    for s, e, v in C.runs(fold):
        if v in ("H", "A"):
            k, o = (0, 1) if v == "H" else (1, 0)
            pt = p25[s:e, k] / (p25[s:e, k] + p25[s:e, o])
            rows.append(((e - s) / 25, np.mean(pred[s:e] == v), np.mean(pt), np.mean(pt > 0.5), np.mean(pred[s:e] == "D")))
    # true team only inside true possessions of at least L seconds; elsewhere stage 1 as is
    sc = [C.spell_f1(fold, pred)]
    for L in (20, 10, 5, 2, 0):
        q = pred.copy()
        for s, e, v in C.runs(fold):
            if v in ("H", "A") and (e - s) / 25 >= L:
                m = np.isin(pred[s:e], ["H", "A"])
                q[s:e][m] = v
        code = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[c] for c in q], np.int8)
        sc.append(C.spell_f1(fold, q))
    return np.array(rows), sc

if __name__ == "__main__":
    names = [os.path.basename(p)[:-4] for p in C.paths()]
    with Pool(4) as pool:
        R = pool.map(one, names)
    print("stage 1, then true team inside true possessions >= 20, 10, 5, 2, 0 s:")
    for row in np.array(C.provider_mean(names, [r[1] for r in R])):
        print("  " + "  ".join(f"{x:.3f}" for x in row))
    for prov in C.PROVIDERS:
        st = np.concatenate([r[0] for n, r in zip(names, R) if n.startswith(prov)])
        print(prov, "true possessions: n, share, frames named right, mean P(true team), frames leaning right, frames called dead; and of those <50% right: n, mean P")
        for lo, hi in BINS:
            m = (st[:, 0] >= lo) & (st[:, 0] < hi)
            w = m & (st[:, 1] < 0.5)
            print(f"  {lo:>3}-{hi if hi < 1e8 else 'inf':>4}  {m.sum():5d} {m.mean():.2f}  {st[m,1].mean():.2f}  {st[m,2].mean():.2f}  {st[m,3].mean():.2f}  {st[m,4].mean():.2f}   | {w.sum():5d} {st[w,2].mean():.2f} dead {st[w,4].mean():.2f}")
