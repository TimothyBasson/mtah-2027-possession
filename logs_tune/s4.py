import numpy as np, tl, os
C = tl.CACHE
a = dict(np.load(os.path.join(C, "loso_f3_300.npz"))); a.update(np.load(os.path.join(C, "loso_sk_f3_300.npz")))
np.savez(os.path.join(C, "loso_f3full.npz"), **a)
cfgs, tags = [], []
for f in ("control_loso.npz", "loso_f3full.npz", "loso_f4_300.npz"):
    for cw in (0.2, 0.3):
        cfgs.append(dict(loso=f, cw=cw)); tags.append("%s cw %g" % (f, cw))
for o, t in zip(tl.eval_configs(cfgs), tags): tl.show(t, o)
o = tl.eval_configs([dict(loso="loso_f6_300.npz")], sets=("skillcorner",))[0]
print("f6 skillcorner", round(o["skillcorner"], 3))
