"""Write the held-out predictions a killed run did not get to: python deep/short/fill.py TAG FOLD"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import data as D
from net import load_net
from train import predict_half, split
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
tag, fold = sys.argv[1:3]
out = os.path.join(D.ROOT, "data", "cache", "deep", tag, fold)
net = load_net(os.path.join(out, "net.pt"), D.NFEAT)
train, val, test = split(D.load_all(), fold)
for kind, ms in (("val", val), ("test", test)):
    for m in ms:
        f = os.path.join(out, m["name"] + ".npz")
        if not os.path.exists(f):
            np.savez_compressed(f, kind=kind, **{f"p{i}": predict_half(net, h) for i, h in enumerate(m["halves"])})
            print(m["name"], flush=True)
