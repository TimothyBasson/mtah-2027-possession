"""The touch detector on BAR-ATH's own ball: a label-free check that it fires as on the simulation.

    python alldata/touch/tbar.py --touch t1 --fold idsse [--public]

Prints touches per minute of seen ball and the score spread, next to the same numbers on the
simulated outside matches. --public lists the detector's touches inside the public five minutes
beside the public labels (a look at the two brief Home touches of the failure review; a check
only, nothing is picked on it).
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import tfeats  # noqa: E402
import tfit  # noqa: E402
import tuse  # noqa: E402

sys.path.append(os.path.join(C.ROOT, "possession"))


def bar_probs(touch, fold):
    """(n, 2) touch probabilities on BAR-ATH, cached per detector."""
    dst = os.path.join(C.TOUCH, "runs", touch, fold, "BAR-ATH.npz")
    import load
    players, ball, team, halves = load.load()
    n = len(ball)
    if os.path.exists(dst):
        return np.load(dst)["q"].astype(np.float32), ball, halves
    committed = os.path.join(HERE, "models", touch + ".pt")  # the all-match detector; else a fresh tfit.py run
    ck = torch.load(committed if fold == "none" and os.path.exists(committed) else os.path.join(C.TOUCH, "runs", touch, fold, "net.pt"), map_location="cpu")
    net = tfit.Net(len(ck["names"]), ck["d"])
    net.load_state_dict(ck["sd"])
    q = np.zeros((n, 2), np.float32)
    for a, b in halves:
        F, names = tfeats.rows(players[a:b + 1], team, ball[a:b + 1, :2].astype(np.float32), load.FPS)
        assert names == ck["names"]
        q[a:b + 1] = tfit.predict(net, F, tfeats.symmetry(names))
    np.savez_compressed(dst, q=q.astype(np.float16))
    return q, ball, halves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--fold", default="idsse")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    q, ball, halves = bar_probs(a.touch, a.fold)
    seen = np.isfinite(ball[:, 0])
    f, tm, sc = tuse.events(q, 0.3)
    mins = sum(seen[x:y + 1].sum() for x, y in halves) / 25 / 60
    print(f"BAR-ATH: {len(f) / mins:.1f} touches per minute of seen ball; score quartiles {np.round(np.percentile(sc, [25, 50, 75]), 2)}; home share {np.mean(tm == 0):.2f}")
    import glob
    for pv in C.PROVIDERS:
        r = []
        for g in sorted(glob.glob(os.path.join(C.TOUCH, "runs", a.touch, pv, pv + "*.npz")))[:4]:
            z = np.load(g)
            fz = np.load(os.path.join(tfeats.OUT, os.path.basename(g)))
            for i in range(len(fz["halves"])):
                for v in (1, 4):
                    e = tuse.events(z[f"q{v}_{i}"], 0.3)
                    r.append((v, len(e[0]) / ((fz[f"F{v}_{i}"][:, 0] > 0.5).sum() / 1500), np.median(e[2])))
        r = np.array(r)
        print(f"  simulated {pv} (held out): " + ", ".join(f"version {v}: {r[r[:, 0] == v, 1].mean():.1f} per minute, median score {r[r[:, 0] == v, 2].mean():.2f}" for v in (1, 4)))
    if a.public:
        import pandas as pd
        gt = pd.read_csv(os.path.join(C.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
        fr, lab = gt["frame"].to_numpy(), gt["label"].to_numpy()
        e = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        s, t = np.r_[0, e], np.r_[e, len(lab)]
        print("public window: true spell (frames, seconds, label) | detector touches inside (offset s, team, score)")
        for x, y in zip(s, t):
            if lab[x] == "X":
                continue
            fa, fb = fr[x], fr[y - 1] + 1
            m = (f >= fa) & (f < fb)
            print(f"  {fa}-{fb} {(fb - fa) / 25:5.1f}s {lab[x]} | " + " ".join(f"{(ff - fa) / 25:.1f}{'HA'[k]}{s_:.2f}" for ff, k, s_ in zip(f[m], tm[m], sc[m])))


if __name__ == "__main__":
    main()
