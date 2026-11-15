import tl, sys, os
tags = sys.argv[1:]
cfgs, names = [], []
for t in tags:
    for cw in (0.2, 0.3, 0.5):
        cfgs.append(dict(loso="loso_%s.npz" % t, cw=cw)); names.append("%s cw %g" % (t, cw))
for o, t in zip(tl.eval_configs(cfgs, sets=("skillcorner",)), names):
    print("{:28s} sk {:.3f}".format(t, o["skillcorner"]), flush=True)
