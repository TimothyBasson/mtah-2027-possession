"""How well a ball-taught net's pointer finds the player on the ball, on held-out matches.

    python deep/short/pointer.py bt [n matches per provider]

Truth: the tracked player nearest the real ball, on live frames where the ball is seen and within
3 m of somebody. Reports top-1 player, right team, and the same split by how long ago the nearest
player last changed team (fresh turnovers are the hard part).
"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import data as D
from net import load_net
torch.set_num_threads(1)
tag = sys.argv[1]; nper = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ms = D.load_all()
sc = np.array([52.5, 34.0], np.float32)
for fold in ("idsse", "metrica", "skillcorner"):
    net = load_net(os.path.join(D.ROOT, "data", "cache", "deep", tag, fold, "net.pt"), D.NFEAT)
    A = []
    for m in [m for m in ms if m["provider"] == fold][:nper]:
        for h in m["halves"]:
            with torch.no_grad():
                _, _, _, ex = net(torch.from_numpy(h["X"])[None], torch.from_numpy(h["M"])[None])
            s = ex["ptr"][0].numpy()
            d = np.linalg.norm(h["X"][..., :2] * sc - (h["ball"] * sc)[:, None], axis=-1)
            d[~h["M"]] = 1e4
            near = np.nan_to_num(d, nan=1e4).argmin(1); dn = np.nan_to_num(d, nan=1e4).min(1)
            ok = np.isfinite(h["ball"][:, 0]) & (h["y"] >= 0) & (h["y"] < 2) & (dn < 3)
            team = h["X"][..., 6]
            tt = team[np.arange(len(near)), near]; pk = s.argmax(1); pt = team[np.arange(len(near)), pk]
            p = torch.softmax(torch.from_numpy(s), -1).numpy(); mass = (p * (team > 0)).sum(1)
            # seconds since the nearest player's team last changed (among ok frames)
            since = np.zeros(len(tt)); last = 0
            for i in range(1, len(tt)):
                if ok[i] and ok[i - 1] and tt[i] != tt[i - 1]: last = i
                since[i] = (i - last) / 5
            A.append(np.stack([ok, pk == near, pt == tt, (mass > 0.5) == (tt > 0), since], 1))
    A = np.concatenate(A); A = A[A[:, 0] > 0]
    line = f"{fold:12s} n {len(A):6d}  top-1 player {A[:, 1].mean():.3f}  pointed team {A[:, 2].mean():.3f}  team by mass {A[:, 3].mean():.3f}   by time since the ball changed team:"
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 1e9)):
        k = (A[:, 4] >= lo) & (A[:, 4] < hi)
        line += f"  {lo}-{hi if hi < 100 else 'inf'}s {A[k, 3].mean():.2f} ({k.mean():.0%})"
    print(line, flush=True)
