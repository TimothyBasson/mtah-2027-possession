"""Train the touch detector with one provider held out; save per-frame touch probabilities.

    python alldata/touch/tfit.py --fold idsse --tag t1 [--seed 0]

A dilated 1-D convolution net over the 25 fps rows of tfeats.py. Three classes per frame: no
touch, home touch, away touch (the recorded ball's touches, +-2 frames). Frames where the shown
ball is unseen are left out of the loss. Writes data/cache/alldata/touch/runs/<tag>/<fold>/<match>.npz
with q<v>_<half>: (n, 2) float16 probabilities of a home / away touch for ball versions 1 and 4,
for the held-out provider's matches (kind test) and the validation matches (kind val). The split
is fit.py's, so a validation match is never trained on.
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
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import tfeats  # noqa: E402

VAL = {"idsse": 1, "metrica": 1, "skillcorner": 3}
EVALV = (1, 2, 3, 4)


class Block(nn.Module):
    def __init__(self, d, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.c2(F.gelu(self.c1(self.norm(x)))))


class Net(nn.Module):
    def __init__(self, nf, d=64, dils=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 32), drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(nf, d, 5, padding=2)
        self.tcn = nn.Sequential(*[Block(d, k, drop) for k in dils])
        self.out = nn.Conv1d(d, 3, 1)

    def forward(self, x):  # (B, T, F) -> (B, T, 3)
        return self.out(self.tcn(self.inp(x.transpose(1, 2)))).transpose(1, 2)


def load(name, versions):
    z = np.load(os.path.join(tfeats.OUT, name + ".npz"))
    nh = len(z["halves"])
    return dict(name=name, provider=name.split("_")[0], names=[str(s) for s in z["names"]],
                halves=[dict(F={v: z[f"F{v}_{i}"] for v in versions}, y=z[f"y_{i}"].astype(np.int64)) for i in range(nh)])


def split(names, fold):
    train, val, test = [], [], []
    for prov in C.PROVIDERS:
        ms = sorted(n for n in names if n.startswith(prov))
        if prov == fold:
            test += ms
        else:
            train += ms[:-VAL[prov]]
            val += ms[-VAL[prov]:]
    return train, val, test


def swap_y(y):
    return np.where(y == 1, 2, np.where(y == 2, 1, y))


@torch.no_grad()
def predict(net, X, sym, chunk=20000, pad=600):
    """(n, 2) home / away touch probability, averaged over the team swap and both mirrors."""
    perm, fx, fy = sym
    net.eval()
    n = len(X)
    out = np.zeros((n, 2), np.float32)
    for a in range(0, n, chunk):
        lo, hi = max(a - pad, 0), min(a + chunk + pad, n)
        Z0 = X[lo:hi].astype(np.float32)
        acc = 0
        for s in (0, 1):
            for m in (0, 1):
                Z = Z0[:, perm] if s else Z0
                if m:
                    Z = Z * fx * fy
                p = F.softmax(net(torch.from_numpy(np.ascontiguousarray(Z))[None])[0], -1).numpy()[:, 1:]
                acc = acc + (p[:, ::-1] if s else p)
        out[a:a + chunk] = (acc / 4)[a - lo:a - lo + min(chunk, n - a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos", type=float, default=5.0, help="weight of touch frames in the loss")
    ap.add_argument("--vers", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--share", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--predict", action="store_true", help="no training: load the saved net and write the probabilities again")
    a = ap.parse_args()
    share = {k: float(v) for k, v in (x.split("=") for x in a.share)}
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(tfeats.OUT, "*.npz")))
    tr, va, te = split(names, a.fold)
    if a.fold == "none":  # every outside match trains: the detector that reads BAR-ATH (tbar.py)
        tr, va = tr + va, []
    train = [load(n, a.vers) for n in (tr[:1] if a.predict else tr)]
    cols = train[0]["names"]
    sym = tfeats.symmetry(cols)
    perm, fx, fy = sym
    provs = sorted({m["provider"] for m in train})
    rest = (1 - sum(share.get(p, 0) for p in provs)) / max(sum(p not in share for p in provs), 1)
    pw = np.array([share.get(p, rest) for p in provs]); pw = pw / pw.sum()
    net = Net(len(cols), a.d)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    wcls = torch.tensor([1.0, a.pos, a.pos])
    out = os.path.join(C.TOUCH, "runs", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if a.predict:
        net.load_state_dict(torch.load(os.path.join(out, "net.pt"), map_location="cpu")["sd"])
    for step in range(0 if a.predict else a.steps):
        net.train()
        Xs, ys = [], []
        while len(Xs) < a.batch:
            prov = provs[rng.choice(len(provs), p=pw)]
            ms = [m for m in train if m["provider"] == prov]
            h = ms[rng.integers(len(ms))]["halves"]
            h = h[rng.integers(len(h))]
            v = a.vers[rng.integers(len(a.vers))]
            s = rng.integers(0, len(h["y"]) - a.T)
            X = h["F"][v][s:s + a.T].astype(np.float32)
            y = h["y"][s:s + a.T].copy()
            y[X[:, 0] < 0.5] = -1
            if (y >= 0).mean() < 0.3:
                continue
            if rng.random() < 0.5:
                X, y = X[:, perm], swap_y(y)
            if rng.random() < 0.5:
                X = X * fx
            if rng.random() < 0.5:
                X = X * fy
            Xs.append(X); ys.append(y)
        X, y = torch.from_numpy(np.stack(Xs)), torch.from_numpy(np.stack(ys))
        loss = F.cross_entropy(net(X).reshape(-1, 3), y.reshape(-1), weight=wcls, ignore_index=-1)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            print(f"{a.tag} {a.fold} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    if not a.predict:
        torch.save(dict(sd=net.state_dict(), names=cols, d=a.d), os.path.join(out, "net.pt"))
    del train
    for kind, group in (("val", va), ("test", te)):
        for n in group:
            m = load(n, EVALV)
            o = {f"q{v}_{i}": predict(net, h["F"][v], sym).astype(np.float16) for i, h in enumerate(m["halves"]) for v in EVALV}
            np.savez_compressed(os.path.join(out, n + ".npz"), kind=kind, **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
