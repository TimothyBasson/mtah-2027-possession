import tl, itertools
cfgs, tags = [], []
def add(tag, **c): cfgs.append(c); tags.append(tag)
add("v8 baseline")
for cw in (0.3, 0.4, 0.5, 0.7): add("cw %g" % cw, cw=cw)
for w in (0.1, 0.15, 0.3): add("weight %g" % w, weight=w)
for p in (8, 16, 20, 30): add("poss %g" % p, poss=p)
for s in (4, 12, 16): add("stop %g" % s, stop=s)
for t in (0.2, 0.3, 0.5): add("to_dead %g" % t, to_dead=t)
for ww in (0.0, 0.1): add("width %g" % ww, width_weight=ww)
for s in (5, 12, 25): add("smooth %d" % s, smooth=s)
for k in ("w_gap", "w_place", "w_speed", "w_flight"):
    for v in (0.0, 0.5, 1.5): add("%s %g" % (k, v), **{k: v})
for o, t in zip(tl.eval_configs(cfgs), tags): tl.show(t, o)
