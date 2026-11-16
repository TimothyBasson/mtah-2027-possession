"""alldata/learned/fit.py with training aimed at brief possessions won under pressure.

    python alldata/contest/cfit.py --fold idsse --tag c1 --extra --brief 3 [fit.py's options]

Same network, folds, outputs and file layout as fit.py (so score.py and breakdown.py read the
runs), plus:

--extra         the contest columns of cfeats.py join the rows (who the ball came off / goes to)
--brief W       frames of a possession shorter than --brief-len seconds that follows the other
                team's possession weigh W times as much in the loss (each head on its own labels);
                not on pseudo-labelled matches
--focal G       hard-example mining inside the loss: a frame's loss is scaled by
                (1 - p of its label) ** G, so frames the net already has right fade out. Frames
                not mined keep 0.5 ** G, so as the net fits, their share of the gradient grows
                and --share no longer holds exactly
--mine-skip P   providers whose frames are not mined (SkillCorner's labels are the noisy ones;
                mining them chases the noise)
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "alldata", "learned"), os.path.join(ROOT, "deep")]
import cfeats  # noqa: E402
import fit as base  # noqa: E402
from fit import CACHE, EVAL, Net, edges, predict, pseudo_labels, split, swap_y  # noqa: E402

EXTRA = os.path.join(CACHE, "contest")


def load_matches(drop_cols=(), folder=base.FEATS, extra=False, extra_dir=EXTRA):
    ms = base.load_matches(drop_cols=drop_cols, folder=folder)
    if extra:
        for m in ms:
            z = np.load(os.path.join(extra_dir, m["name"] + ".npz"))
            for i, h in enumerate(m["halves"]):
                h["F"] = [np.concatenate([f, z[f"F{v}_{i}"]], 1) for v, f in enumerate(h["F"])]
            m["names"] = m["names"] + [str(n) for n in z["names"]]
    return ms


def brief_weights(y, w, max_rows):
    """w on the frames of H/A possessions shorter than max_rows that follow the other team."""
    out = np.ones(len(y), np.float32)
    if w == 1:
        return out
    e = np.flatnonzero(np.diff(y) != 0) + 1
    s, t = np.r_[0, e], np.r_[e, len(y)]
    for j in range(1, len(s)):
        k = y[s[j]]
        if k in (0, 1) and y[s[j] - 1] == 1 - k and t[j] - s[j] < max_rows:
            out[s[j]:t[j]] = w
    return out


def batch(train, rng, B, T, a, sym):
    perm, sw, mi = sym
    provs = sorted({m["provider"] for m in train})
    fixed = {k: v for k, v in a.share.items() if k in provs}
    rest = (1 - sum(fixed.values())) / max(len(provs) - len(fixed), 1)
    pw = np.array([fixed.get(k, rest) for k in provs])
    Xs, ys, es, ws, y2s, w2s, mine = [], [], [], [], [], [], []
    rows = int(round(a.brief_len * 25 / cfeats.STEP))
    while len(Xs) < B:
        prov = provs[rng.choice(len(provs), p=pw / pw.sum()) if fixed else rng.integers(len(provs))]
        ms = [m for m in train if m["provider"] == prov]
        m = ms[rng.integers(len(ms))]
        h = m["halves"][rng.integers(len(m["halves"]))]
        if rng.random() < a.sim and len(h["F"]) > 1:
            v = a.vers[rng.integers(len(a.vers))] if a.vers else rng.integers(1, min(len(h["F"]), 4))
        else:
            v = 0
        yall = h["y"]
        n = len(yall)
        s = rng.integers(0, max(1, n - T))
        X, y, y2 = h["F"][v][s:s + T], yall[s:s + T], h["yf"][s:s + T]
        if len(X) < T or (y >= 0).mean() < 0.3:
            continue
        # pseudo-labelled matches are not reweighted: their brief spells are the teacher's own
        # guesses, and weighting them makes the adapted net cut finer still (review of PR 61)
        bw = 1.0 if prov == "pseudo" else a.brief
        key = ("bw", bw, rows)
        if key not in h:
            h[key] = brief_weights(yall, bw, rows), brief_weights(h["yf"], bw, rows)
        w, w2 = h[key][0][s:s + T], h[key][1][s:s + T]
        if rng.random() < 0.5:
            X, y, y2 = X[:, perm] * sw, swap_y(y), swap_y(y2)
        if rng.random() < 0.5:
            X = X * mi
        Xs.append(X.astype(np.float32)); ys.append(y); es.append(edges(y)); ws.append(w); y2s.append(y2); w2s.append(w2)
        mine.append(np.full(T, prov not in a.mine_skip, np.float32))
    t = lambda v: torch.from_numpy(np.stack(v))  # noqa: E731
    return t(Xs), t(ys), t(es), t(ws), t(y2s), t(w2s), t(mine)


def head_loss(logits, y, w, mine, gamma):
    ok = y >= 0
    lg, yy, ww = logits[ok], y[ok], w[ok]
    ce = F.cross_entropy(lg, yy, reduction="none", label_smoothing=0.02)
    if gamma > 0:
        with torch.no_grad():
            pt = F.softmax(lg, -1).gather(1, yy[:, None]).squeeze(1)
            f = torch.where(mine[ok] > 0, (1 - pt) ** gamma, torch.full_like(pt, 0.5 ** gamma))
        ww = ww * f
    return (ce * ww).sum() / ww.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sim", type=float, default=1.0)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--share", nargs="*", default=[])
    ap.add_argument("--vers", nargs="*", type=int, default=[])
    ap.add_argument("--pv", type=int, default=1)
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--brief", type=float, default=1.0)
    ap.add_argument("--brief-len", type=float, default=5.0)
    ap.add_argument("--focal", type=float, default=0.0)
    ap.add_argument("--mine-skip", nargs="*", default=["skillcorner", "pseudo"])
    a = ap.parse_args()
    a.share = {k: float(v) for k, v in (x.split("=") for x in a.share)}
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ms = load_matches(a.drop_cols, extra=a.extra)
    train, val, test = split(ms, a.fold)
    if a.fold == "none":
        train, val = train + val, []
        test = load_matches(a.drop_cols, os.path.join(CACHE, "bar"), a.extra, os.path.join(CACHE, "contest_bar"))
    names = ms[0]["names"]
    if a.pseudo:
        train = train + [pseudo_labels(m, a.pseudo, a.fold, a.conf, a.pv) for m in test]
    sym = cfeats.symmetry(names)
    net = Net(len(names), a.d)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(CACHE, "runs", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for step in range(a.steps):
        net.train()
        X, y, e, w, y2, w2, mine = batch(train, rng, a.batch, a.T, a, sym)
        logits, edge = net(X)
        ok = y >= 0
        loss = (head_loss(logits[..., :3], y, w, mine, a.focal) + head_loss(logits[..., 3:], y2, w2, mine, a.focal)
                + a.edge * F.binary_cross_entropy_with_logits(edge[ok], e[ok]))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            print(f"{a.tag} {a.fold} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(sd=net.state_dict(), names=names, d=a.d), os.path.join(out, "net.pt"))
    for kind, group in (("val", val), ("test", test)):
        for m in group:
            o = {}
            for i, h in enumerate(m["halves"]):
                for v, k in EVAL.items():
                    if v < len(h["F"]):
                        q = predict(net, h["F"][v], sym)
                        o[f"{k}{i}"], o[f"f{k}{i}"] = q[:, :3], q[:, 3:]
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind, **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
