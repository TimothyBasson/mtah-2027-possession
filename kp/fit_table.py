"""Keeper table counted on all three providers (each provider equal), as frames per million,
for hmm/decode.py's KEEPER_GIVEN_HOME / KEEPER_GIVEN_DEAD.  python kp/fit_table.py box 01"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_keeper as sk
ends, cats = sys.argv[1], sys.argv[2]
paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(sk.ROOT, "data", "external", "*.npz")))}
with Pool(4, initializer=sk._init, initargs=(dict(paths=paths, configs=[(ends, cats)]),)) as pool:
    cnt = dict(pool.map(sk.count_one, list(paths), chunksize=1))
h = d = 0
for k in sk.sh.SETS:
    hh = sum(cnt[n][(ends, cats)][0] for n in cnt if n.startswith(k))
    dd = sum(cnt[n][(ends, cats)][1] for n in cnt if n.startswith(k))
    h = h + hh / hh.sum() * 1e6
    d = d + dd / dd.sum() * 1e6
print("home", np.round(h, 1).tolist())
print("dead", np.round(d, 1).tolist())
t = sk.keeper.table(h, d)
print("log p(.|H)-log p(.|D)", np.round(t[0] - t[2], 2))
