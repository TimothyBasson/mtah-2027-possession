"""Frozen features of a tag's fold nets for every outside match, for the heads in this folder.

    python deep/spell/extract.py --tag v2 --fold idsse

Writes data/cache/spell/<tag>/<fold>/<match>.npz: per half and per symmetric view (team swap x
mirror), the 128-d state the net's three heads read (float16) and its H/A/D logits. Players only:
the net never sees the ball.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data as D  # noqa: E402
from net import PossessionNet  # noqa: E402


@torch.no_grad()
def hidden(net, X, M):
    x, m = torch.from_numpy(X)[None], torch.from_numpy(M)[None]
    B, T, K, Fd = x.shape
    h = torch.cat([net.enc(x[0, i:i + 4000], m[0, i:i + 4000]) for i in range(0, T, 4000)])[None]
    h = net.tcn(h.transpose(1, 2)).transpose(1, 2)
    if net.gru is not None:
        h = h + net.gru(h)[0]
    return h[0].numpy(), net.cls(h)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    src = os.path.join(D.ROOT, "data", "cache", "deep", a.tag, a.fold)
    out = os.path.join(D.ROOT, "data", "cache", "spell", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    net = PossessionNet(nfeat=D.NFEAT)
    net.load_state_dict(torch.load(os.path.join(src, "net.pt")))
    net.eval()
    dummy = None
    for m in D.load_all():
        f = os.path.join(out, m["name"] + ".npz")
        if os.path.exists(f):
            continue
        z = {}
        for i, h in enumerate(m["halves"]):
            dummy = np.zeros(len(h["y"]), np.int64)
            for s in (0, 1):
                for mi in (0, 1):
                    X = h["X"]
                    if s:
                        X, _, _ = D.swap_teams(X, dummy, h["ball"])
                    if mi:
                        X, _ = D.mirror(X, h["ball"])
                    hid, lg = hidden(net, X, h["M"])
                    z[f"h{i}_{s}{mi}"] = hid.astype(np.float16)
                    z[f"l{i}_{s}{mi}"] = lg.astype(np.float32)
        np.savez(f + ".tmp.npz", **z)
        os.replace(f + ".tmp.npz", f)
        print(a.fold, m["name"], flush=True)


if __name__ == "__main__":
    main()
