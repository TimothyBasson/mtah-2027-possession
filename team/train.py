"""Train the team-naming net with one provider held out, and save per-frame probabilities.

    python team/train.py --fold idsse --tag t1

The net is deep/net.py's (set encoder, convolutions, ball-taught pointer, GRU) on team/data.py's
tokens, trained for one job: on frames where the ball is live under the folded labels, which team
has it. The three-way H/A/D loss is kept at a low weight so the net still knows about stoppages
(who restarts tells who had it), but its alive/dead output is not what this step is for. Output
layout is deep/train.py's, so deep/decode.py and team/apply.py read it:
data/cache/deep/<tag>/<fold>/<match>.npz, H/A/D at 5 fps.
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
import tdata as TD  # noqa: E402
import data as D  # noqa: E402
from net import PossessionNet  # noqa: E402
from train import ball_losses, ball_targets, edges, split  # noqa: E402  (deep/train.py)


def batch(train, rng, B, T, a):
    provs = sorted({m["provider"] for m in train})
    Xs, Ms, ys, bs, es = [], [], [], [], []
    while len(Xs) < B:
        prov = provs[rng.integers(len(provs))]
        ms = [m for m in train if m["provider"] == prov]
        h = ms[rng.integers(len(ms))]["halves"]
        h = h[rng.integers(len(h))]
        s = rng.integers(0, max(1, len(h["y"]) - T))
        X, M, y, ball = h["X"][s:s + T].astype(np.float32), h["M"][s:s + T].copy(), h["y"][s:s + T], h["ball"][s:s + T]
        if len(X) < T or (y >= 0).mean() < 0.3:
            continue
        e = edges(y)
        if rng.random() < 0.5:
            X, ball = TD.swap_teams(X), -ball
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
        if rng.random() < 0.5:
            X, ball = TD.mirror(X), ball * np.array([1, -1], np.float32)
        if a.noise:
            X[..., :2] += rng.normal(0, a.noise / 40, X[..., :2].shape).astype(np.float32)
            seen = np.repeat(X[..., D.NFEAT + 2 * len(TD.OFFSETS):], 2, -1)
            X[..., TD.PATH] += rng.normal(0, 0.03, X[..., TD.PATH].shape).astype(np.float32) * seen
        M &= ~(rng.random(M.shape) < rng.uniform(0, a.dropout))
        X[~M] = 0
        Xs.append(X); Ms.append(M); ys.append(y); bs.append(ball.astype(np.float32)); es.append(e)
    t = lambda v: torch.from_numpy(np.stack(v))  # noqa: E731
    return t(Xs), t(Ms), t(ys), t(bs), t(es)


@torch.no_grad()
def predict_half(net, X, M):
    """(T, 3) probabilities, averaged over the four symmetric views."""
    net.eval()
    X = X.astype(np.float32)
    out = 0
    for s in (0, 1):
        for m in (0, 1):
            V = X
            if s:
                V = TD.swap_teams(V)
            if m:
                V = TD.mirror(V)
            p = F.softmax(net(torch.from_numpy(V)[None], torch.from_numpy(M)[None])[0][0], -1).numpy()
            out = out + (p[:, [1, 0, 2]] if s else p)
    return out / 4


def team_acc(net, ms):
    r, n = 0, 0
    for m in ms:
        for h in m["halves"]:
            p = predict_half(net, h["X"], h["M"])
            g = (h["y"] == 0) | (h["y"] == 1)
            r += ((p[g, 1] > p[g, 0]) == (h["y"][g] == 1)).sum(); n += g.sum()
    return r / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--three", type=float, default=0.3, help="weight of the H/A/D loss beside the team loss")
    ap.add_argument("--no-path", action="store_true", help="control: stage 1's tokens, this training")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    matches = TD.load_all()
    nf = D.NFEAT if a.no_path else TD.NFEAT
    if a.no_path:
        for m in matches:
            for h in m["halves"]:
                h["X"] = h["X"][..., :nf]
    train, val, test = split(matches, a.fold) if a.fold != "none" else (matches, [], [])
    net = PossessionNet(nfeat=nf, ptr=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(D.ROOT, "data", "cache", "deep", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0, run, start = time.time(), [], 0
    ckpt = os.path.join(out, "ckpt.pt")
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"])
        start = c["step"]
        print(f"resumed at step {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        net.train()
        X, M, y, ball, e = batch(train, rng, a.batch, a.window, a)
        logits, bhat, ehat, extra = net(X, M)
        g = y >= 0
        live = (y == 0) | (y == 1)
        loss_t = F.binary_cross_entropy_with_logits(logits[live][:, 1] - logits[live][:, 0], (y[live] == 1).float())
        loss_c = F.cross_entropy(logits[g], y[g])
        bg = live & torch.isfinite(ball[..., 0])
        loss_b = F.smooth_l1_loss(bhat[bg], ball[bg], beta=0.1) if bg.any() else logits.sum() * 0
        loss_e = F.binary_cross_entropy_with_logits(ehat[g], e[g])
        bl = ball_losses(extra, ball_targets(X, M, ball, y))
        loss = (loss_t + a.three * loss_c + 0.5 * loss_b + 0.3 * loss_e
                + 0.5 * bl["ptr"] + 0.3 * bl["bvel"] + 0.3 * bl["touch"] + 0.2 * bl["free"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append((loss_t.item(), loss_c.item(), loss_b.item(), bl["ptr"].item()))
        if step % 100 == 0 or step == a.steps:
            r = np.mean(run, 0); run = []
            msg = f"{step:5d} team {r[0]:.3f} hadx {r[1]:.3f} ball {r[2]:.3f} ptr {r[3]:.3f}  {time.time() - t0:.0f}s"
            if step % 500 == 0 or step == a.steps:
                msg += f"  team acc val {team_acc(net, val):.3f} test {team_acc(net, test[:3]):.3f}"
            print(msg, flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), step=step), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    for kind, ms in (("val", val), ("test", test)):
        for m in ms:
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind,
                                **{f"p{i}": predict_half(net, h["X"], h["M"]) for i, h in enumerate(m["halves"])})


if __name__ == "__main__":
    main()
