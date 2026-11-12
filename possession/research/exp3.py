import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab
from lab import *
ms = load()
W = (list(ms[0]["who"]), ["s1", "s3", "d2"])
# dead designs can mix views per feature: build manually
def dead_run(spec, tag):
    """spec: list of (names, views) blocks concatenated."""
    Xs = None
    for names, views in spec:
        X, _ = design(ms, "dead", names, views)
        Xs = X if Xs is None else [np.c_[a, b] for a, b in zip(Xs, X)]
    key = ("dead", ("custom", tag), ("x",))
    design.__defaults__[0][key] = [(x, ["c"]) for x in Xs]
    return run(ms, who=W, dead=(("custom", tag), ("x",)), tag=tag)
B = (["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])
V = ["s1", "s3", "d2"]
dead_run([B], "dead base")
for nm in ms[0]["dead"]:
    if nm in ("sp_med", "acc_p80"): continue
    dead_run([B, ([nm], V)], f"  + {nm}")
rest = [n for n in ms[0]["dead"] if n not in ("sp_med", "acc_p80")]
dead_run([B, (rest, V)], "all dead features")
dead_run([(list(ms[0]["dead"]), ["s0.5", "s1", "s3", "s6", "d2", "d5"])], "all dead, 6 views")
