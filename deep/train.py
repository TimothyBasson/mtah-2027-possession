"""Train the learned possession model with one provider held out, and save per-frame probabilities.

    python deep/train.py --fold idsse [--steps 3000] [--tag v2]

Trains on the other two providers (each drawn equally often), minus a few of their matches kept
back as validation for choosing decode settings (deep/decode.py). Writes
data/cache/deep/<tag>/<fold>/<match>.npz with H/A/D probabilities at 5 fps for the held-out
provider's matches and the validation matches. CPU only.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import trackers as TR  # noqa: E402
from net import PossessionNet  # noqa: E402

VAL = {"idsse": 1, "metrica": 1, "skillcorner": 3, "soccertrack": 1}  # last n matches of each training provider
PAD = 50  # frames of context either side of a raw window, for smoothing and velocities


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
    """1 where the label changes within 2 steps (0.4 s) either side; ungraded frames don't count."""
    ch = np.zeros(len(y), bool)
    d = (y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)
    for k in range(-2, 3):
        i = np.flatnonzero(d) + 1 + k
        ch[i[(i >= 0) & (i < len(y))]] = True
    return ch.astype(np.float32)


def spell_weights(y, power, ref=25, lo=0.5, hi=4.0):
    """Per-frame loss weight so short possessions are not drowned by long ones: a frame in an H or A
    run of n steps weighs (ref / n) ** power, clipped. spellF1 counts each possession once, a
    per-frame loss counts it once per frame. ref is 5 s at 5 fps."""
    w = np.ones(len(y), np.float32)
    if power <= 0:
        return w
    edge = np.flatnonzero(np.diff(y) != 0) + 1
    for s, e in zip(np.r_[0, edge], np.r_[edge, len(y)]):
        if y[s] in (0, 1):
            w[s:e] = np.clip((ref / (e - s)) ** power, lo, hi)
    return w


def batch(train, rng, B, T, a):
    provs = sorted({m["provider"] for m in train})
    Xs, Ms, ys, bs, es, ws = [], [], [], [], [], []
    while len(Xs) < B:
        if a.share:  # e.g. soccertrack=0.15: that provider's share of windows, the rest split equally
            w = np.array([a.share.get(q, np.nan) for q in provs])
            w[np.isnan(w)] = (1 - np.nansum(w)) / max(1, np.isnan(w).sum())
            prov = provs[rng.choice(len(provs), p=w / w.sum())]
        else:
            prov = provs[rng.integers(len(provs))]
        ms = [m for m in train if m["provider"] == prov]
        m = ms[rng.integers(len(ms))]
        hi = rng.integers(len(m["halves"]))
        # the same half with players lost in runs and bursts, as BAR-ATH's tracker loses them
        h = (m["knocked"] if "knocked" in m and rng.random() < a.knock else m["halves"])[hi]
        n = len(h["y"])
        s = rng.integers(0, max(1, n - T))
        X, M, y, ball = h["X"][s:s + T], h["M"][s:s + T], h["y"][s:s + T], h["ball"][s:s + T]
        f0, f1 = s * D.STEP - PAD, (s + T) * D.STEP + PAD
        if "raw" in m and rng.random() < a.style and f0 >= 0 and f1 <= len(m["raw"][hi]["xy"]):
            # the same window as another tracking system might have recorded it (deep/trackers.py)
            r = m["raw"][hi]
            X, M = D.window_tensors(TR.apply(r["xy"][f0:f1], TR.style(rng), rng, (0.0, 0.0), r["fps"]), r["team"], r["fps"])
            X, M = X[PAD // D.STEP:PAD // D.STEP + T], M[PAD // D.STEP:PAD // D.STEP + T]
        if "w" not in h or h.get("wp") != a.spell:
            h["w"], h["wp"] = spell_weights(h["y"], a.spell), a.spell
        ws.append(h["w"][s:s + T])
        if len(X) < T or (y >= 0).mean() < 0.3:
            ws.pop()
            continue
        e = edges(y)
        if rng.random() < 0.5:
            X, y, ball = D.swap_teams(X, y, ball)
        if rng.random() < 0.5:
            X, ball = D.mirror(X, ball)
        X, M = X.copy(), M.copy()
        if a.noise:
            X[..., :2] += rng.normal(0, a.noise / 40, X[..., :2].shape).astype(np.float32)
        p = rng.uniform(0, a.dropout)
        drop = rng.random(M.shape) < p
        M &= ~drop
        X[~M] = 0
        Xs.append(X); Ms.append(M); ys.append(y); bs.append(ball); es.append(e)
    t = lambda v: torch.from_numpy(np.stack(v))  # noqa: E731
    return t(Xs), t(Ms), t(ys), t(bs), t(es), t(ws)


@torch.no_grad()
def predict_half(net, h, tta=True):
    """(T, 3) probabilities, averaged over the four symmetric views of the half."""
    net.eval()
    out = 0
    views = [(s, m) for s in (0, 1) for m in (0, 1)] if tta else [(0, 0)]
    dummy_y, dummy_b = np.zeros(len(h["y"]), np.int64), h["ball"]
    for s, m in views:
        X = h["X"]
        if s:
            X, _, _ = D.swap_teams(X, dummy_y, dummy_b)
        if m:
            X, _ = D.mirror(X, dummy_b)
        logits = net(torch.from_numpy(X)[None], torch.from_numpy(h["M"])[None])[0][0]
        p = F.softmax(logits, -1).numpy()
        out = out + (p[:, [1, 0, 2]] if s else p)
    return out / len(views)


def frame_scores(net, ms):
    acc, n = 0, 0
    for m in ms:
        for h in m["halves"]:
            p = predict_half(net, h, tta=False)
            g = h["y"] >= 0
            acc += (p.argmax(1)[g] == h["y"][g]).sum(); n += g.sum()
    return acc / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.2, help="position jitter, metres")
    ap.add_argument("--dropout", type=float, default=0.15, help="max share of players hidden per window")
    ap.add_argument("--ball", type=float, default=0.5, help="weight of the where-is-the-ball loss")
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--spell", type=float, default=0.0, help="power of the short-possession loss weight")
    ap.add_argument("--knock", type=float, default=0.0,
                    help="share of training windows drawn with players lost in runs, as on BAR-ATH")
    ap.add_argument("--style", type=float, default=0.0,
                    help="share of training windows changed to look like another tracking system")
    ap.add_argument("--with", dest="extra", nargs="*", default=[],
                    help="providers beyond the three of v2 to train on as well, e.g. soccertrack "
                         "(data/soccertrack/); without it the training set is v2's whatever is on disk")
    ap.add_argument("--share", nargs="*", default=[], help="provider=share of training windows, e.g. soccertrack=0.15")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.share = {k: float(v) for k, v in (kv.split("=") for kv in a.share)}
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    matches = D.load_all()
    if a.knock:
        for m, k in zip(matches, D.load_all(knock=True)):
            assert m["name"] == k["name"]
            m["knocked"] = k["halves"]
    matches = [m for m in matches if m["provider"] in ("idsse", "metrica", "skillcorner") or m["provider"] in a.extra]
    train, val, test = split(matches, a.fold) if a.fold != "none" else (matches, [], [])
    if a.style:
        for m in train:
            m["raw"] = D.load_raw(D.match_path(m["name"]))
    net = PossessionNet(nfeat=D.NFEAT, d=a.d, gru=not a.no_gru)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(D.ROOT, "data", "cache", "deep", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0, run, start = time.time(), [], 0
    ckpt = os.path.join(out, "ckpt.pt")
    if os.path.exists(ckpt):  # the cloud container can restart mid-run: carry on from the last save
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]
        if "torch_rng" in c:  # dropout draws from torch's generator: without it a resumed run diverges
            torch.set_rng_state(c["torch_rng"])
        start = c["step"]
        print(f"resumed at step {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        net.train()
        X, M, y, ball, e, w = batch(train, rng, a.batch, a.window, a)
        logits, bhat, ehat = net(X, M)
        g = y >= 0
        loss_c = (F.cross_entropy(logits[g], y[g], reduction="none") * w[g]).sum() / w[g].sum()
        bg = g & (y < 2) & torch.isfinite(ball[..., 0])
        loss_b = F.smooth_l1_loss(bhat[bg], ball[bg], beta=0.1) if bg.any() else logits.sum() * 0
        loss_e = F.binary_cross_entropy_with_logits(ehat[g], e[g])
        loss = loss_c + a.ball * loss_b + a.edge * loss_e
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append((loss_c.item(), loss_b.item(), loss_e.item()))
        if step % 100 == 0 or step == a.steps:
            r = np.mean(run, 0); run = []
            msg = f"{step:5d} ce {r[0]:.3f} ball {r[1]:.3f} edge {r[2]:.3f}  {time.time() - t0:.0f}s"
            if step % 500 == 0 or step == a.steps:
                msg += f"  val acc {frame_scores(net, val):.3f}  test acc {frame_scores(net, test[:3]):.3f}"
            print(msg, flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), step=step), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    for kind, ms in (("val", val), ("test", test)):
        for m in ms:
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind,
                                **{f"p{i}": predict_half(net, h) for i, h in enumerate(m["halves"])})


if __name__ == "__main__":
    main()
