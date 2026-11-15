import tl
cfgs, tags = [], []
def add(tag, **c): cfgs.append(c); tags.append(tag)
add("base")
for v in ("raw", "folded"):
    for s in ("pair", "all"):
        add("tabs %s %s" % (v, s), tabs=(v, s))
        add("tabs %s %s, v6" % (v, s), tabs=(v, s), cw=0)
for o, t in zip(tl.eval_configs(cfgs), tags): tl.show(t, o)
