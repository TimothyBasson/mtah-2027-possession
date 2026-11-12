import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
old = list(ms[0]["who"])
for m, p in zip(ms, paths):
    m["who"].update(feats2.cached(p))
new = [k for k in ms[0]["who"] if k not in old]
V = ["s1", "s3", "d2"]
base = ["wid", "len", "vcx"]
run(ms, who=(base, V), tag="base")
for nm in new:
    run(ms, who=(base + [nm], V), tag=f"  + {nm}")
run(ms, who=(base + new, V), tag="base + all new")
run(ms, who=(old + new, V), tag="all old + all new")
