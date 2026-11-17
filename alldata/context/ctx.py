"""Whole-match context pass: a second stage that re-reads the first stage's held-out output over a
whole half, so each brief segment is judged with minutes of context around it.

    python alldata/context/ctx.py --fold idsse --tag cx1 [--seed 0] [--base pw1e] [--touch t1]

Inputs per 5 fps frame (all on the BAR-ATH-like ball, version 1):
  - the first stage's probabilities from the fold where the match's own provider was held out
    (run --base, keys s / fs: labels as built and folded), as log-probabilities;
  - the learned touch detector's probabilities (run --touch, the fold that held the provider out),
    max-pooled from 25 fps to 5 fps;
  - the first stage's own rows (alldata/learned/build.py, version 1: ball, players against it, bt).
Model: frame encoder and a short TCN (local), then a transformer over the whole half in 2 s tokens with
absolute match-time encoding (long-range: team styles, field-position trends, who wins the ball
where), upsampled and added back; heads give a residual on the first stage's log-probabilities
(zero at the start, so training starts from the first stage). Trained on whole halves of the
other providers' matches, SkillCorner a quarter of draws (Gambtt's rule). Writes runs/<tag>/<fold>/
<match>.npz with s* / fs* for the held-out provider's matches (kind test), so score.py and
alldata/touch/tuse.py read them like any first-stage run.
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned")]
import feats  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "alldata")
PROVS = ("idsse", "metrica", "skillcorner")


def load(base, touch, use_rows=True, ver=1):
    ms = []
    for f in sorted(glob.glob(os.path.join(CACHE, "feats", "*.npz"))):
        name = os.path.basename(f)[:-4]
        prov = name.split("_")[0]
        if prov not in PROVS:
            continue
        z = np.load(f)
        names = [str(n) for n in z["names"]]
        r = np.load(os.path.join(CACHE, "runs", base, prov, name + ".npz"))
        assert str(r["kind"]) == "test"
        tq = np.load(os.path.join(CACHE, "touch", "runs", touch, prov, name + ".npz")) if touch else None
        halves = []
        for i in range(int(z["nhalves"])):
            Fr = z[f"F{ver}_{i}"]
            T = len(Fr)
            p = np.log(np.clip(r[f"s{i}"][:T], 1e-4, 1)).astype(np.float32)
            pf = np.log(np.clip(r[f"fs{i}"][:T], 1e-4, 1)).astype(np.float32)
            cols = [p, pf]
            if tq is not None:
                q = tq[f"q{ver}_{i}"]
                n5 = min(T, len(q) // 5 + 1)
                qq = np.zeros((T, 2), np.float32)
                pad = np.zeros((n5 * 5, 2), np.float32)
                pad[:len(q)] = q[:n5 * 5]
                qq[:n5] = pad.reshape(n5, 5, 2).max(1)
                cols.append(qq)
            if use_rows:
                cols.append(Fr.astype(np.float32))
            halves.append(dict(X=np.concatenate(cols, 1), base=np.concatenate([p, pf], 1),
                               y=z[f"y_{i}"].astype(np.int64), yf=z[f"yf_{i}"].astype(np.int64)))
        ms.append(dict(name=name, provider=prov, halves=halves))
    cn = ["lp_H", "lp_A", "lp_D", "lpf_H", "lpf_A", "lpf_D"] + (["tq_H", "tq_A"] if touch else []) + (names if use_rows else [])
    return ms, cn


def symmetry(cn):
    perm, sw, mi = feats.symmetry(cn)
    pos = {n: i for i, n in enumerate(cn)}
    for a, b in (("lp_H", "lp_A"), ("lpf_H", "lpf_A"), ("tq_H", "tq_A")):
        if a in pos:
            perm[pos[a]], perm[pos[b]] = pos[b], pos[a]
    return perm, sw, mi


class TCNBlock(nn.Module):
    def __init__(self, d, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.c2(F.gelu(self.c1(self.norm(x)))))


def time_enc(n, d, offset=0):
    """Sinusoidal encoding of the token index since the half started (2 s tokens)."""
    t = torch.arange(n, dtype=torch.float32)[:, None] + offset
    k = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    e = torch.zeros(n, d)
    e[:, 0::2], e[:, 1::2] = torch.sin(t * k), torch.cos(t * k)
    return e


class Net(nn.Module):
    def __init__(self, nf, d=64, layers=3, heads=4, drop=0.1, pool=10, ctx=True, team_only=False):
        super().__init__()
        self.pool, self.ctx = pool, ctx
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.tcn = nn.Sequential(*[TCNBlock(d, k, drop) for k in (1, 2, 4, 8, 1, 2, 4, 8)])
        if ctx:
            el = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(el, layers)
            self.mix = nn.Linear(2 * d, d)
        self.post = nn.Sequential(*[TCNBlock(d, k, drop) for k in (1, 4, 16)])
        self.team_only = team_only
        self.out = nn.Linear(d, 2 if team_only else 6)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, base):
        B, T, _ = x.shape
        h = self.inp(x)
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        if self.ctx:
            P = self.pool
            Tp = (T + P - 1) // P
            hp = F.pad(h, (0, 0, 0, Tp * P - T)).reshape(B, Tp, P, -1).mean(2)
            hp = hp + time_enc(Tp, hp.shape[-1]).to(hp)
            c = self.enc(hp)
            c = c.repeat_interleave(P, 1)[:, :T]
            h = h + self.mix(torch.cat([h, c], -1))
        h = self.post(h.transpose(1, 2)).transpose(1, 2)
        o = self.out(h)
        if self.team_only:
            # only the H/A split moves: +delta/2 on H, -delta/2 on A; the first stage's dead
            # probability is kept exactly (log-sum of H and A is unchanged up to the split)
            out = []
            for k, b in ((0, base[..., :3]), (1, base[..., 3:])):
                lh, la, ld = b[..., 0], b[..., 1], b[..., 2]
                alive = torch.logaddexp(lh, la)
                z = (lh - la) + o[..., k]
                out.append(torch.stack([alive + F.logsigmoid(z), alive + F.logsigmoid(-z), ld], -1))
            return torch.cat(out, -1)
        return base + o  # (B, T, 6) logits: residual on the first stage's log-probs


def swap_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def draw(train, rng, B, share, sym, crop, noise):
    perm, sw, mi = sym
    provs = sorted({m["provider"] for m in train})
    fixed = {k: v for k, v in share.items() if k in provs}
    rest = (1 - sum(fixed.values())) / max(len(provs) - len(fixed), 1)
    pw = np.array([fixed.get(k, rest) for k in provs]); pw /= pw.sum()
    out = []
    for _ in range(B):
        prov = provs[rng.choice(len(provs), p=pw)]
        ms = [m for m in train if m["provider"] == prov]
        h = ms[rng.integers(len(ms))]["halves"][rng.integers(2)]
        T = len(h["y"])
        L = int(T * rng.uniform(crop, 1.0))
        s = rng.integers(0, T - L + 1)
        s -= s % 10   # keep the 2 s token grid aligned with the half's start
        X, bs, y, yf = h["X"][s:s + L], h["base"][s:s + L], h["y"][s:s + L], h["yf"][s:s + L]
        if rng.random() < 0.5:
            X = X[:, perm] * sw
            bs = bs[:, [1, 0, 2, 4, 3, 5]]
            y, yf = swap_y(y), swap_y(yf)
        if rng.random() < 0.5:
            X = X * mi
        if noise:
            X = X + rng.normal(0, noise, X.shape).astype(np.float32)
        out.append((X, bs, y, yf))
    L = max(len(o[0]) for o in out)
    nf = out[0][0].shape[1]
    Xb = np.zeros((B, L, nf), np.float32); Bb = np.zeros((B, L, 6), np.float32)
    Yb = np.full((B, L), -1, np.int64); Yf = np.full((B, L), -1, np.int64)
    for i, (X, bs, y, yf) in enumerate(out):
        n = len(X)
        Xb[i, :n], Bb[i, :n], Yb[i, :n], Yf[i, :n] = X, bs, y, yf
    t = torch.from_numpy
    return t(Xb), t(Bb), t(Yb), t(Yf)


@torch.no_grad()
def predict(net, h, sym):
    perm, sw, mi = sym
    net.eval()
    out = 0
    for s in (0, 1):
        for m in (0, 1):
            X, bs = h["X"], h["base"]
            if s:
                X, bs = X[:, perm] * sw, bs[:, [1, 0, 2, 4, 3, 5]]
            if m:
                X = X * mi
            lg = net(torch.from_numpy(np.ascontiguousarray(X))[None], torch.from_numpy(np.ascontiguousarray(bs))[None])[0]
            p = torch.cat([F.softmax(lg[:, :3], -1), F.softmax(lg[:, 3:], -1)], -1).numpy()
            out = out + (p[:, [1, 0, 2, 4, 3, 5]] if s else p)
    return out / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="pw1e")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--no-rows", action="store_true")
    ap.add_argument("--no-ctx", action="store_true", help="control: same net without the whole-half transformer")
    ap.add_argument("--team-only", action="store_true", help="keep the first stage's dead probability; only re-split H/A")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--share", nargs="*", default=["skillcorner=0.25"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    share = {k: float(v) for k, v in (x.split("=") for x in a.share)}
    assert share.get("skillcorner", 0) <= 0.25, "SkillCorner stays at a quarter share at most"
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ms, cn = load(a.base, a.touch, not a.no_rows)
    train = [m for m in ms if m["provider"] != a.fold]
    test = [m for m in ms if m["provider"] == a.fold]
    sym = symmetry(cn)
    net = Net(len(cn), a.d, a.layers, ctx=not a.no_ctx, team_only=a.team_only)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(CACHE, "runs", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for step in range(a.steps):
        net.train()
        X, bs, y, yf = draw(train, rng, a.batch, share, sym, a.crop, a.noise)
        lg = net(X, bs)
        loss = (F.cross_entropy(lg[..., :3].reshape(-1, 3), y.reshape(-1), ignore_index=-1, label_smoothing=0.02)
                + F.cross_entropy(lg[..., 3:].reshape(-1, 3), yf.reshape(-1), ignore_index=-1, label_smoothing=0.02))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"{a.tag} {a.fold} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(sd=net.state_dict(), names=cn, d=a.d, layers=a.layers, ctx=not a.no_ctx, team_only=a.team_only), os.path.join(out, "net.pt"))
    for m in test:
        o = {}
        for i, h in enumerate(m["halves"]):
            q = predict(net, h, sym)
            o[f"s{i}"], o[f"fs{i}"] = q[:, :3], q[:, 3:]
        np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind="test", **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
