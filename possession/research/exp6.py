"""Odd interactions: an antisymmetric 'who' feature times a symmetric context, and x|x|."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
allw = list(ms[0]["who"])
Xw, names = design(ms, "who", allw, ["s1", "s3", "d2"])
key = ["wid", "len", "vcx", "speed", "press_max", "spread", "pol", "nn_opp_back", "vor", "front3_speed", "keeper_speed", "field_x"]
K, _ = design(ms, "who", key, ["s1", "s3"])
C, cn = design(ms, "dead", ["sp_med", "spread_sum", "len_sum", "nn_opp_all", "pol_sum"], ["s1"])
def put(tag, Xs):
    k = ("who", ("custom", tag), ("x",)); design.__defaults__[0][k] = [(x, ["c"]) for x in Xs]
    return run(ms, who=(("custom", tag), ("x",)), tag=tag)
put("all linear", Xw)
def z(c):  # centre contexts roughly so the product is an interaction, not a copy
    return c - np.nanmedian(c, 0)
put("+ x|x| on 12 key (s1,s3)", [np.c_[x, k * np.abs(k)] for x, k in zip(Xw, K)])
inter = [np.column_stack([k[:, i] * z(c)[:, j] for i in range(0, k.shape[1], 2) for j in range(c.shape[1])]) for k, c in zip(K, C)]
put("+ key(s1) x 5 contexts", [np.c_[x, i] for x, i in zip(Xw, inter)])
# field position magnitude as a context: how far up the pitch play is
F, _ = design(ms, "who", ["field_x"], ["s1"])
put("+ key(s1) x |field_x|", [np.c_[x, k[:, ::2] * np.abs(f)] for x, k, f in zip(Xw, K, F)])
put("+ all three", [np.c_[x, k * np.abs(k), i, k[:, ::2] * np.abs(f)] for x, k, i, f in zip(Xw, K, inter, F)])
