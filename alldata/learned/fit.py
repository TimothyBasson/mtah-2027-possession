"""Train the all-data possession model with one provider held out; save per-frame probabilities.

    python alldata/learned/fit.py --fold idsse --tag b1 [--seed 0] [--pseudo b1]

Inputs are the rows of feats.py (ball, players against the ball, the players-only net's
probabilities). Training draws each provider equally often, and each window with the ball either
as recorded or under simulated BAR-ATH-like visibility (--sim: share of windows simulated).
Writes data/cache/alldata/runs/<tag>/<fold>/<match>.npz: probabilities at 5 fps for the held-out
provider (kind test) and the validation matches (kind val), with the ball as recorded (p*) and
under the simulation's first seed (s*); f* are the head taught on the folded labels.

--pseudo TAG adapts to the unseen provider without its labels: the held-out matches join the
training set labelled by run TAG's own decoded predictions (frames where it is unsure are left
out of the loss). With --fold none the unseen match is BAR-ATH (run.py --features first).
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
sys.path[:0] = [HERE, os.path.join(ROOT, "deep")]
import feats  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "alldata")
FEATS = os.path.join(CACHE, "feats")  # build.py's match files
VAL = {"idsse": 1, "metrica": 1, "skillcorner": 3, "soccertrack": 1}


class TCNBlock(nn.Module):
    def __init__(self, d, dilation, drop):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.c2(F.gelu(self.c1(self.norm(x)))))


class Net(nn.Module):
    def __init__(self, nf, d=96, dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32), drop=0.1, gru=True):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.tcn = nn.Sequential(*[TCNBlock(d, k, drop) for k in dilations])
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True) if gru else None
        self.cls = nn.Linear(d, 6)  # H/A/D on the labels as built, then on the folded labels
        self.edge = nn.Linear(d, 1)

    def forward(self, x):
        h = self.inp(x)
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        if self.gru is not None:
            h = h + self.gru(h)[0]
        return self.cls(h), self.edge(h).squeeze(-1)


def load_matches(drop_cols=(), folder=FEATS):
    ms = []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        z = np.load(f)
        name = os.path.basename(f)[:-4]
        names = [str(n) for n in z["names"]]
        keep = [i for i, n in enumerate(names) if not any(n.startswith(c) for c in drop_cols)]
        nv = sum(1 for k in z.files if k.endswith("_0") and k[0] == "F")
        halves = []
        for i in range(int(z["nhalves"])):
            halves.append(dict(F=[z[f"F{v}_{i}"][:, keep] for v in range(nv)], y=z[f"y_{i}"].astype(np.int64),
                               yf=z[f"yf_{i}"].astype(np.int64), fr=z[f"fr_{i}"]))
        ms.append(dict(name=name, provider=name.split("_")[0], halves=halves, names=[names[i] for i in keep]))
    return ms


def pseudo_labels(m, tag, fold, conf):
    """The match with y / yf replaced by run `tag`'s decoded labels; -1 where its probability of
    the decoded state is under conf."""
    sys.path.append(os.path.join(ROOT, "deep"))
    import decode
    z = np.load(os.path.join(CACHE, "runs", tag, fold, m["name"] + ".npz"))
    halves = []
    for i, h in enumerate(m["halves"]):
        new = dict(F=h["F"][:1], fr=h["fr"])
        for key, pk in (("y", "p"), ("yf", "fp")):
            p = z[f"{pk}{i}"]
            lp = np.log(np.clip(p, 1e-6, 1))
            path = decode.absorb(decode.viterbi(lp, 4 / 5), {0: 5, 1: 5, 2: 5}).astype(np.int64)
            path[p[np.arange(len(p)), path] < conf] = -1
            new[key] = path
        halves.append(new)
    return dict(name=m["name"], provider="pseudo", halves=halves, names=m["names"])


def split(matches, fold):
    train, val, test = [], [], []
    for prov in sorted({m["provider"] for m in matches}):
        ms = [m for m in matches if m["provider"] == prov]
        if prov == fold:
            test += ms
        else:
            train += ms[:-VAL[prov]]
            val += ms[-VAL[prov]:]
    return train, val, test


def edges(y):
    ch = np.zeros(len(y), bool)
    d = (y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)
    for k in range(-2, 3):
        i = np.flatnonzero(d) + 1 + k
        ch[i[(i >= 0) & (i < len(y))]] = True
    return ch.astype(np.float32)


def spell_weights(y, power, ref=25, lo=0.5, hi=4.0):
    w = np.ones(len(y), np.float32)
    if power <= 0:
        return w
    e = np.flatnonzero(np.diff(y) != 0) + 1
    for s, t in zip(np.r_[0, e], np.r_[e, len(y)]):
        if y[s] in (0, 1):
            w[s:t] = np.clip((ref / (t - s)) ** power, lo, hi)
    return w


def swap_y(y):
    return np.where(y == 0, 1, np.where(y == 1, 0, y))


def batch(train, rng, B, T, a, sym):
    perm, sw, mi = sym
    provs = sorted({m["provider"] for m in train})
    Xs, ys, es, ws, y2s = [], [], [], [], []
    while len(Xs) < B:
        prov = provs[rng.integers(len(provs))]
        ms = [m for m in train if m["provider"] == prov]
        m = ms[rng.integers(len(ms))]
        h = m["halves"][rng.integers(len(m["halves"]))]
        v = rng.integers(1, len(h["F"])) if (rng.random() < a.sim and len(h["F"]) > 1) else 0
        yall = h["y"]
        n = len(yall)
        s = rng.integers(0, max(1, n - T))
        X, y, y2 = h["F"][v][s:s + T], yall[s:s + T], h["yf"][s:s + T]
        if len(X) < T or (y >= 0).mean() < 0.3:
            continue
        key = ("w", a.spell)
        if key not in h:
            h[key] = spell_weights(yall, a.spell)
        w = h[key][s:s + T]
        if rng.random() < 0.5:
            X, y, y2 = X[:, perm] * sw, swap_y(y), swap_y(y2)
        if rng.random() < 0.5:
            X = X * mi
        if a.noise:
            X = X + rng.normal(0, a.noise, X.shape).astype(np.float32)
        Xs.append(X.astype(np.float32)); ys.append(y); es.append(edges(y)); ws.append(w); y2s.append(y2)
    t = lambda v: torch.from_numpy(np.stack(v))  # noqa: E731
    return t(Xs), t(ys), t(es), t(ws), t(y2s)


@torch.no_grad()
def predict(net, X, sym):
    """(T, 3) probabilities, averaged over the four symmetric views."""
    perm, sw, mi = sym
    net.eval()
    out = 0
    for s in (0, 1):
        for m in (0, 1):
            Z = X
            if s:
                Z = Z[:, perm] * sw
            if m:
                Z = Z * mi
            lg = net(torch.from_numpy(np.ascontiguousarray(Z, np.float32))[None])[0][0]
            p = torch.cat([F.softmax(lg[:, :3], -1), F.softmax(lg[:, 3:], -1)], -1).numpy()
            out = out + (p[:, [1, 0, 2, 4, 3, 5]] if s else p)
    return out / 4  # (T, 6): labels as built, folded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)   # idsse | metrica | skillcorner | none
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--spell", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ms = load_matches(drop_cols=a.drop_cols)
    train, val, test = split(ms, a.fold)
    if a.fold == "none":  # every outside match trains; the unseen match is BAR-ATH (run.py --features)
        train, val = train + val, []
        test = load_matches(drop_cols=a.drop_cols, folder=os.path.join(CACHE, "bar"))
    names = ms[0]["names"]
    if a.pseudo:
        train = train + [pseudo_labels(m, a.pseudo, a.fold, a.conf) for m in test]
    sym = feats.symmetry(names)
    net = Net(len(names), a.d)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(CACHE, "runs", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for step in range(a.steps):
        net.train()
        X, y, e, w, y2 = batch(train, rng, a.batch, a.T, a, sym)
        logits, edge = net(X)
        ok, ok2 = y >= 0, y2 >= 0
        ce = F.cross_entropy(logits[ok][:, :3], y[ok], reduction="none", label_smoothing=0.02)
        ce2 = F.cross_entropy(logits[ok2][:, 3:], y2[ok2], reduction="none", label_smoothing=0.02)
        loss = ((ce * w[ok]).sum() / w[ok].sum() + (ce2 * w[ok2]).sum() / w[ok2].sum()
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
                for k, v in (("p", 0), ("s", 1)):
                    if v < len(h["F"]):
                        q = predict(net, h["F"][v], sym)
                        o[f"{k}{i}"], o[f"f{k}{i}"] = q[:, :3], q[:, 3:]
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind, **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
