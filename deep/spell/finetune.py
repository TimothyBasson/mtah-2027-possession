"""Carry on training a whole fold net, through the minimum-duration chain or per frame.

    python deep/spell/finetune.py --fold idsse --name ft_crf --loss crf --margin 2
    python deep/spell/finetune.py --fold idsse --name ft_ce --loss ce          # the control

Starts from data/cache/deep/<tag>/<fold>/net.pt and trains every weight for --steps more on the
control-rule labels, with deep/train.py's windows and augmentation and its ball target. --loss crf
is head.py's sequence loss reaching all the way down; --loss ce is the same budget spent on the
per-frame loss, so that "more training" and "other labels" are not mistaken for the objective.
Scores the held-out provider as head.py does.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import data as D  # noqa: E402
import train as T  # noqa: E402
from crf import MinDurCRF  # noqa: E402
from head import decode_chain, decode_frames, project  # noqa: E402
from net import PossessionNet  # noqa: E402


@torch.no_grad()
def scores_half(net, h, probs):
    """(T, 3) class scores (or probabilities) averaged over the four symmetric views."""
    net.eval()
    out = 0
    dummy = np.zeros(len(h["y"]), np.int64)
    for s in (0, 1):
        for m in (0, 1):
            X = h["X"]
            if s:
                X, _, _ = D.swap_teams(X, dummy, h["ball"])
            if m:
                X, _ = D.mirror(X, h["ball"])
            e = net(torch.from_numpy(X)[None], torch.from_numpy(h["M"])[None])[0][0]
            e = (F.softmax(e, -1) if probs else e).numpy()
            out = out + (e[:, C.SWAP] if s else e)
    return out / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--loss", default="crf", choices=("ce", "crf"))
    ap.add_argument("--mins", default="10,10,5")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--spell", type=float, default=0.5, help="power of the per-possession weight on the margin")
    ap.add_argument("--scales", default="1,2,3,5")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.mins = tuple(int(v) for v in a.mins.split(","))
    a.scales = tuple(float(v) for v in a.scales.split(","))
    aug = argparse.Namespace(knock=0.0, spell=a.spell, noise=0.2, dropout=0.15)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    matches = D.load_all(control=True)
    labels = {m["name"]: m for m in C.labels_all()}
    train, val, test = T.split(matches, a.fold)
    if a.loss == "crf":
        for m in train:
            for h in m["halves"]:
                h["y"] = project(h["y"], a.mins)
    out = os.path.join(C.ROOT, "data", "cache", "spell", a.tag, a.fold, "ft")
    os.makedirs(out, exist_ok=True)
    net = PossessionNet(nfeat=D.NFEAT)
    net.load_state_dict(torch.load(os.path.join(C.ROOT, "data", "cache", "deep", a.tag, a.fold, "net.pt")))
    crf = MinDurCRF(a.mins) if a.loss == "crf" else None
    params = list(net.parameters()) + (list(crf.parameters()) if crf else [])
    opt = torch.optim.AdamW(params, a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    ckpt, start = os.path.join(out, f"{a.name}_s{a.seed}.ckpt"), 0
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        if crf:
            crf.load_state_dict(c["crf"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"]); start = c["step"]
    t0, run = time.time(), []
    for step in range(start + 1, a.steps + 1):
        net.train()
        X, M, y, ball, e, w = T.batch(train, rng, a.batch, a.window, aug)
        logits, bhat, _ = net(X, M)
        g = y >= 0
        if crf:
            loss_c = crf.nll(logits, y, a.margin * w if a.margin else None).sum() / g.sum()
        else:
            loss_c = F.cross_entropy(logits[g], y[g])
        bg = g & (y < 2) & torch.isfinite(ball[..., 0])
        loss_b = F.smooth_l1_loss(bhat[bg], ball[bg], beta=0.1) if bg.any() else logits.sum() * 0
        loss = loss_c + 0.5 * loss_b
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run.append(loss_c.item())
        if step % 50 == 0 or step == a.steps:
            print(f"  {a.fold} {a.name} {step:5d} loss {np.mean(run):.4f}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            run = []
            torch.save(dict(net=net.state_dict(), crf=crf.state_dict() if crf else None, opt=opt.state_dict(),
                            sched=sched.state_dict(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            step=step), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    ms = [(kind, m) for kind, group in (("val", val), ("test", test)) for m in group]
    Es = [[scores_half(net, h, crf is None) for h in m["halves"]] for _, m in ms]
    best = None
    for k in (a.scales if crf else (1.0,)):
        res = {}
        flat = decode_chain(crf, [(k * e).astype(np.float32) for E2 in Es for e in E2]) if crf else \
            [decode_frames(P, 4, 2, 1) for E2 in Es for P in E2]
        for (kind, m), E2 in zip(ms, Es):
            paths, flat = flat[:len(E2)], flat[len(E2):]
            for view in ("raw", "control"):
                res.setdefault((kind, m["provider"], view), []).append(C.spell_f1(labels[m["name"]], paths, view))
        vals = {v: np.mean([np.mean(x) for (kd, p, vw), x in res.items() if kd == "val" and vw == v]) for v in ("raw", "control")}
        tests = {v: np.mean(res[("test", a.fold, v)]) for v in ("raw", "control")}
        print(f"  {a.fold} {a.name} scale {k:g} test raw {tests['raw']:.4f} control {tests['control']:.4f} "
              f"val raw {vals['raw']:.4f} control {vals['control']:.4f}", file=sys.stderr, flush=True)
        if best is None or vals["control"] > best[1]["control"]:
            best = (k, vals, tests)
    k, vals, tests = best
    torch.save(dict(net=net.state_dict(), crf=crf.state_dict() if crf else None, scale=k, args=vars(a)),
               os.path.join(out, f"{a.name}_s{a.seed}.pt"))
    print(f"RESULT {a.name} seed {a.seed} {a.fold} test raw {tests['raw']:.4f} control {tests['control']:.4f} "
          f"val raw {vals['raw']:.4f} control {vals['control']:.4f} scale {k:g}", flush=True)


if __name__ == "__main__":
    main()
