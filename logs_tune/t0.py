import time, tl
t = time.time()
o = tl.eval_configs([dict(cw=0)])
tl.show("v6", o[0]); print(time.time() - t)
