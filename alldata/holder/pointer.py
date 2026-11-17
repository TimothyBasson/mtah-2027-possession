"""How well does the pointer name the holder and the team on the held-out provider?

    python alldata/holder/pointer.py --tag k1

Per fold, on the held-out matches with the BAR-ATH-like ball (version 1), frames where a holder
can be read off the ball as recorded: how often the pointer's top player is that holder, against
"the player nearest the ball the network is shown"; and on all live frames how often the
pointer's heavier team is the labelled team, against "the team of the nearest player" and the
network's own H/A output.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import hfit as H  # noqa: E402

B, feats = H.B, H.feats

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="k1")
a = ap.parse_args()
torch.set_num_threads(1)
ms = H.attach_players(B.load_matches(drop_cols=["tim"]))
for fold in ("idsse", "metrica", "skillcorner"):
    ck = torch.load(os.path.join(B.CACHE, "runs", a.tag, fold, "net.pt"), map_location="cpu")
    net = H.HolderNet(len(ck["names"]), ck["names"], ck["d"], ck["dp"])
    net.load_state_dict(ck["sd"])
    net.eval()
    acc = np.zeros(7)
    for m in B.split(ms, fold)[2]:
        for h in m["halves"]:
            t = lambda v: torch.from_numpy(np.ascontiguousarray(v))[None]  # noqa: E731
            with torch.no_grad():
                X, P, team = t(h["F"][1]), t(h["P"].astype(np.float32)), t(h["team"])
                lg, _, lp, lm, sel = net(X, P, team)
                tok, ok, _, side = net.tokens(X, P, team)
            lp, lm, sel, lg = lp[0].numpy(), lm[0].numpy(), sel[0].numpy(), lg[0].numpy()
            d = np.where(ok[0].numpy(), np.expm1(tok[0, :, :, 2].numpy() * 2), 1e4)
            top, near = lp.argmax(1), d.argmin(1)
            r = np.arange(len(top))
            y, hold = h["y"], h["hold"]
            held, live = hold >= 0, (y >= 0) & (y <= 1)
            n = sel.shape[1] // 2
            acc += [held.sum(), (sel[r, top] == hold)[held].sum(), (sel[r, near] == hold)[held].sum(), live.sum(),
                    (lm.argmax(1) == y)[live].sum(), ((near >= n) == y)[live].sum(), (lg[:, :2].argmax(1) == y)[live].sum()]
    print(f"{fold:12s} holder: pointer {acc[1] / acc[0]:.3f}  nearest {acc[2] / acc[0]:.3f}   "
          f"team on live frames: pointer {acc[4] / acc[3]:.3f}  nearest {acc[5] / acc[3]:.3f}  network {acc[6] / acc[3]:.3f}", flush=True)
