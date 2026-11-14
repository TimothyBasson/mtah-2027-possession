from lib import *
from e1 import absorb
from hmm.model import rate_log_trans
import sys
F = sys.argv[1]; L = loso(F)
CFG = {}
for w in (.15, .2, .3):
    for ab in (0, 1, 1.5, 2):
        CFG[f"w{w} ab{ab}"] = (w, ab, None)
CFG["w.2 ab1.5 poss20"] = (.2, 1.5, (20, 8, .4)); CFG["w.2 ab1.5 dead.6"] = (.2, 1.5, (12, 8, .6)); CFG["w.2 ab1.5 poss16 dead.5"] = (.2, 1.5, (16, 8, .5))
def one(n):
    m = load(n); r = ratio(L, n); out = {}
    for k, (w, ab, tr) in CFG.items():
        out[k] = sf1(m["folded"], decode(m, w * r, rate_log_trans(tr[0]*25, tr[1]*25, tr[2]) if tr else None, absorb(ab) if ab else None))
    return out
if __name__ == "__main__":
    R = run_all(one)
    for k in CFG: report({n: R[n][k] for n in R}, F[:-4] + " " + k)
