"""Self-training on the unlabelled target match, held out by provider (or on BAR-ATH with --fold none).

    python alldata/selftrain/selftrain.py --fold idsse --tag sa1 --teacher pw1e [--init TAG] [options]

Same inputs, network, source batches and outputs as alldata/learned/fit.py (whose --pseudo is one
round of hard, decoded pseudo-labels). What is new here, all label-free on the target:

  --teacher TAG   soft targets: the saved probabilities of run TAG on the target matches (e.g. the
                  two-seed ensemble pw1e), sharpened by --temp and weighted by their confidence;
                  frames under --conf are left out. Both heads (labels as built, folded).
  --ema A         mean teacher: an exponential moving average of the student also predicts the
                  target window; the target is (1 - A) * fixed teacher + A * EMA teacher.
  --ent W         entropy minimisation on the student's target predictions.
  --strong        the student sees a strongly augmented view of the target window (feature noise
                  scaled per column, column dropout, time masking) while the targets come from the
                  clean view: consistency regularisation. Every augmentation is one BAR-ATH allows
                  (no second ball version: BAR-ATH has one ball).
  --init TAG      test-time adaptation: start from run TAG's fold net and fine-tune (use a small
                  --lr and few --steps) instead of training from scratch.

Writes data/cache/alldata/runs/<tag>/<fold>/ exactly like fit.py, so score.py and ens.py work.
"""

import argparse
import copy
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
import fit  # noqa: E402

SWAP = [1, 0, 2]


def target_halves(test, teacher, fold, pv):
    """[(X (T, nf), q (T, 6))] for every half of every target match: X in the ball version the
    match is scored in, q the teacher's probabilities (labels as built, folded)."""
    out = []
    for m in test:
        z = np.load(os.path.join(fit.CACHE, "runs", teacher, fold, m["name"] + ".npz"))
        v = pv if len(m["halves"][0]["F"]) > 1 else 0
        k = fit.EVAL[v]
        for i, h in enumerate(m["halves"]):
            q = np.concatenate([z[f"{k}{i}"], z[f"f{k}{i}"]], 1).astype(np.float32)
            out.append((h["F"][v].astype(np.float32), q))
    return out


def target_batch(halves, rng, B, T, sym):
    perm, sw, mi = sym
    Xs, qs = [], []
    while len(Xs) < B:
        X, q = halves[rng.integers(len(halves))]
        s = rng.integers(0, max(1, len(X) - T))
        X, q = X[s:s + T], q[s:s + T]
        if len(X) < T:
            continue
        if rng.random() < 0.5:
            X, q = X[:, perm] * sw, q[:, SWAP + [3 + j for j in SWAP]]
        if rng.random() < 0.5:
            X = X * mi
        Xs.append(X); qs.append(q)
    return torch.from_numpy(np.stack(Xs)), torch.from_numpy(np.stack(qs))


def strong(X, colstd, rng, a):
    """Feature noise scaled per column, column dropout and time masking."""
    B, T, nf = X.shape
    X = X + torch.randn_like(X) * colstd * a.noise_s
    keep = (torch.rand(B, 1, nf) > a.coldrop).float()
    X = X * keep
    for b in range(B):
        for _ in range(2):
            L = int(rng.integers(5, 15))   # 1 to 3 s
            s = int(rng.integers(0, T - L))
            X[b, s:s + L] = 0
    return X


def soft_ce(logits, q, w):
    """sum over heads of the confidence-weighted cross-entropy to soft targets q (B, T, 6)."""
    loss = 0
    for h in (0, 1):
        lp = F.log_softmax(logits[..., 3 * h:3 * h + 3], -1)
        ce = -(q[..., 3 * h:3 * h + 3] * lp).sum(-1)
        ww = w[..., h]
        loss = loss + (ce * ww).sum() / ww.sum().clamp(min=1)
    return loss


def entropy(logits):
    e = 0
    for h in (0, 1):
        p = F.softmax(logits[..., 3 * h:3 * h + 3], -1)
        e = e - (p * torch.log(p.clamp(min=1e-8))).sum(-1).mean()
    return e


def sharpen(q, temp):
    out = []
    for h in (0, 1):
        x = q[..., 3 * h:3 * h + 3].clamp(min=1e-6) ** (1 / temp)
        out.append(x / x.sum(-1, keepdim=True))
    return torch.cat(out, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=22, help="source windows a step")
    ap.add_argument("--tbatch", type=int, default=10, help="target windows a step")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--const-lr", action="store_true")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the target loss")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--strong", action="store_true")
    ap.add_argument("--noise-s", type=float, default=0.3)
    ap.add_argument("--coldrop", type=float, default=0.1)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--drop-cols", nargs="*", default=["tim"])
    ap.add_argument("--share", nargs="*", default=["skillcorner=0.25"])
    ap.add_argument("--pv", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.share = {k: float(v) for k, v in (x.split("=") for x in a.share)}
    a.sim, a.vers, a.spell, a.noise = 1.0, [], 0.0, 0.0
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed + 1000)
    ms = fit.load_matches(drop_cols=a.drop_cols)
    train, val, test = fit.split(ms, a.fold)
    if a.fold == "none":
        train, val = train + val, []
        test = fit.load_matches(drop_cols=a.drop_cols, folder=os.path.join(fit.CACHE, "bar"))
    names = ms[0]["names"]
    sym = feats.symmetry(names)
    tgt = target_halves(test, a.teacher, a.fold, a.pv)
    colstd = torch.from_numpy(np.concatenate([X for X, _ in tgt]).std(0).astype(np.float32) + 1e-3)
    net = fit.Net(len(names), a.d)
    if a.init:
        sd = torch.load(os.path.join(fit.CACHE, "runs", a.init, a.fold, "net.pt"))
        net.load_state_dict(sd["sd"])
    ema = copy.deepcopy(net) if a.ema > 0 else None
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0) if a.const_lr else
             torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1))
    out = os.path.join(fit.CACHE, "runs", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for step in range(a.steps):
        net.train()
        X, y, e, w, y2 = fit.batch(train, rng, a.batch, a.T, a, sym)
        Xt, qt = target_batch(tgt, rng, a.tbatch, a.T, sym)
        if ema is not None:
            ema.eval()
            with torch.no_grad():
                lg = ema(Xt)[0]
                qe = torch.cat([F.softmax(lg[..., :3], -1), F.softmax(lg[..., 3:], -1)], -1)
            qt = (1 - a.ema) * qt + a.ema * qe
        conf = torch.stack([qt[..., :3].max(-1).values, qt[..., 3:].max(-1).values], -1)
        wt = conf * (conf >= a.conf).float()
        qs = sharpen(qt, a.temp)
        Xs = strong(Xt, colstd, rng, a) if a.strong else Xt
        logits, edge = net(torch.cat([X, Xs]))
        ls, lt = logits[:len(X)], logits[len(X):]
        ok, ok2 = y >= 0, y2 >= 0
        ce = F.cross_entropy(ls[ok][:, :3], y[ok], reduction="none", label_smoothing=0.02)
        ce2 = F.cross_entropy(ls[ok2][:, 3:], y2[ok2], reduction="none", label_smoothing=0.02)
        loss = ((ce * w[ok]).sum() / w[ok].sum() + (ce2 * w[ok2]).sum() / w[ok2].sum()
                + a.edge * F.binary_cross_entropy_with_logits(edge[:len(X)][ok], e[ok]))
        if a.lam:
            loss = loss + a.lam * soft_ce(lt, qs, wt)
        if a.ent:
            loss = loss + a.ent * entropy(lt)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(a.ema_decay).add_(pn, alpha=1 - a.ema_decay)
        if step % 250 == 0 or step == a.steps - 1:
            print(f"{a.tag} {a.fold} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(sd=net.state_dict(), names=names, d=a.d), os.path.join(out, "net.pt"))
    for kind, group in (("val", val), ("test", test)):
        for m in group:
            o = {}
            for i, h in enumerate(m["halves"]):
                for v, k in fit.EVAL.items():
                    if v < len(h["F"]):
                        q = fit.predict(net, h["F"][v], sym)
                        o[f"{k}{i}"], o[f"f{k}{i}"] = q[:, :3], q[:, 3:]
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind, **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
