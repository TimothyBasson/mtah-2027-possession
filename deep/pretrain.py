"""Pretrain the possession network on player movement alone: no labels and no ball.

    python deep/pretrain.py --tag pre [--steps 2000] [--exclude idsse]

The task is to say where hidden players are. In each one-minute window a few tracks are hidden
for 2 to 30 s each, and sometimes every player is hidden for 1 to 3 s; from what is left the
network predicts, for every frame and each team, where on the pitch that team's hidden players
stand (a distribution over GX x GY cells). Hiding a track for seconds asks for the shape of the
team around the gap; hiding everyone for a moment asks for how play carries on.

Nothing here reads a label or the ball, so the windows are drawn from every match we hold,
BAR-ATH included, and from the provider a fold holds out: that provider's movement is treated as
BAR-ATH's is, seen without labels. One pretrained net therefore serves all three folds
(train.py --init). --exclude leaves a provider's movement out, the control for whether seeing it
is what helps.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(D_ROOT := os.path.dirname(HERE), "possession"), D_ROOT]
import data as D  # noqa: E402
from net import PossessionNet  # noqa: E402

GX, GY = 30, 20  # cells of 3.5 m x 3.4 m


def load_tracks():
    """[{name, provider, halves: [(X (T, K, NFEAT), M (T, K))]}]: every match's players with each
    track in its own column. Players only: the ball that half_tensors returns is dropped here."""
    cache = os.path.join(D.ROOT, "data", "cache", "deep_tracks_v1.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    out = []
    for p in D.external_paths():
        d = np.load(p)
        hs = []
        for a, b in d["halves"]:
            X, M, _, _ = D.half_tensors(d["players"], d["ball"], d["team"], int(a), int(b),
                                        tuple(d["pitch"]), int(d["fps"]), tracks=True)
            hs.append((X, M))
        out.append(dict(name=os.path.basename(p)[:-4], provider=D.provider(p), halves=hs))
    import load
    players, ball, team, halves = load.load()
    hs = []
    for a, b in halves:
        X, M, _, _ = D.half_tensors(players, ball, team, a, b, load.PITCH, load.FPS, tracks=True)
        hs.append((X, M))
    out.append(dict(name="bar_ath", provider="barath", halves=hs))
    with open(cache + f".{os.getpid()}", "wb") as f:
        pickle.dump(out, f, protocol=4)
    os.replace(cache + f".{os.getpid()}", cache)
    return out


def hide_tracks(M, rng):
    """(T, K) bool: which tracked player-frames to hide."""
    T, K = M.shape
    hide = np.zeros((T, K), bool)
    present = np.flatnonzero(M.any(0))
    for k in rng.choice(present, min(len(present), rng.integers(1, 5)), replace=False):
        n = rng.integers(10, 151)
        s = rng.integers(0, T - n + 1)
        hide[s:s + n, k] = True
    for _ in range(rng.integers(0, 3)):
        n = rng.integers(5, 16)
        s = rng.integers(0, T - n + 1)
        hide[s:s + n] = True
    return hide & M


def window(matches, rng, T, noise=0.2, dropout=0.15):
    """One training window: (X seen, M seen, positions (T, SLOTS, 2), team, hidden (T, SLOTS))."""
    provs = sorted({m["provider"] for m in matches})
    while True:
        prov = provs[rng.integers(len(provs))]
        ms = [m for m in matches if m["provider"] == prov]
        X, M = ms[rng.integers(len(ms))]["halves"][rng.integers(2)]
        s = rng.integers(0, max(1, len(X) - T))
        X, M = X[s:s + T], M[s:s + T]
        if len(X) == T and M.any(1).mean() > 0.5:
            break
    keep = np.sort(np.argsort(-M.sum(0), kind="stable")[:D.SLOTS])  # the window's most-seen tracks
    X, M = X[:, keep].copy(), M[:, keep].copy()
    if X.shape[1] < D.SLOTS:
        X = np.pad(X, ((0, 0), (0, D.SLOTS - X.shape[1]), (0, 0)))
        M = np.pad(M, ((0, 0), (0, D.SLOTS - M.shape[1])))
    dummy = np.zeros((T, 2), np.float32)
    if rng.random() < 0.5:
        X, _, _ = D.swap_teams(X, np.zeros(T, np.int64), dummy)
    if rng.random() < 0.5:
        X, _ = D.mirror(X, dummy)
    pos, team = X[..., :2].copy(), X[..., 6].copy()
    hid = hide_tracks(M, rng)
    if noise:
        X[..., :2] += rng.normal(0, noise / 40, X[..., :2].shape).astype(np.float32)
    seen = M & ~hid & ~(rng.random(M.shape) < rng.uniform(0, dropout))
    X[~seen] = 0
    return X, seen, pos, team, hid


def batch(matches, rng, B, T):
    ws = [window(matches, rng, T) for _ in range(B)]
    return [torch.from_numpy(np.stack(v)) for v in zip(*ws)]


def hidden_loss(logits, pos, team, hid):
    """Cross-entropy between the predicted map (B, T, 2 * GX * GY) and where each team's hidden
    players are, over the frames where that team has someone hidden."""
    B, T = hid.shape[:2]
    cx = ((pos[..., 0] + 1) / 2 * GX).long().clamp(0, GX - 1)
    cy = ((pos[..., 1] + 1) / 2 * GY).long().clamp(0, GY - 1)
    cell = cx * GY + cy + (team < 0) * (GX * GY)
    tgt = torch.zeros(B, T, 2 * GX * GY).scatter_add_(2, cell, hid.float()).reshape(B, T, 2, -1)
    n = tgt.sum(-1)
    lp = F.log_softmax(logits.reshape(B, T, 2, -1), -1)
    ce = -(tgt * lp).sum(-1) / n.clamp(min=1)
    return ce[n > 0].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pre")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exclude", default="", help="providers left out, comma-separated")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    matches = [m for m in load_tracks() if m["provider"] not in a.exclude.split(",")]
    print("providers:", sorted({m["provider"] for m in matches}), len(matches), "matches", flush=True)
    net = PossessionNet(nfeat=D.NFEAT)
    head = nn.Linear(128, 2 * GX * GY)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(D.ROOT, "data", "cache", "deep", a.tag, "pretrain")
    os.makedirs(out, exist_ok=True)
    ckpt, start, run, t0 = os.path.join(out, "ckpt.pt"), 0, [], time.time()
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); head.load_state_dict(c["head"])
        opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"])
        start = c["step"]
        print(f"resumed at step {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        net.train()
        X, M, pos, team, hid = batch(matches, rng, a.batch, a.window)
        loss = hidden_loss(head(net.features(X, M)), pos, team, hid)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if step % 50 == 0 or step == a.steps:
            print(f"{step:5d} hidden {np.mean(run):.3f}  {time.time() - t0:.0f}s", flush=True); run = []
            torch.save(dict(net=net.state_dict(), head=head.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), step=step), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    torch.save(head.state_dict(), os.path.join(out, "head.pt"))


if __name__ == "__main__":
    main()
