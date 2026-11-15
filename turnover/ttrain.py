"""Train the turnover detector with one provider held out; save its per-frame event probabilities.

    python turnover/ttrain.py --fold idsse [--tag t1]

Same split as deep/train.py (the other two providers train, their last matches are validation).
Writes data/cache/turnover/<tag>/<fold>/<match>.npz: ev{i}, evf{i}, cls{i} (T, 3) at 12.5 fps per
half for the held-out and validation matches. The ball is a training target only.
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
from tnet import TurnNet  # noqa: E402

VAL = {"idsse": 1, "metrica": 1, "skillcorner": 3}
ROOT = TD.ROOT


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


def widen(ev, r):
    """Each event marked on r frames either side as well."""
    out = ev.copy()
    for k in range(1, r + 1):
        out[k:] = np.where(out[k:] == 0, ev[:-k], out[k:])
        out[:-k] = np.where(out[:-k] == 0, ev[k:], out[:-k])
    return out


def batch(train, rng, B, T, a):
    provs = sorted({m["provider"] for m in train})
    out = {k: [] for k in ("X", "M", "y", "ev", "evf", "ball", "touch")}
    while len(out["X"]) < B:
        prov = provs[rng.integers(len(provs))]
        ms = [m for m in train if m["provider"] == prov]
        h = ms[rng.integers(len(ms))]["halves"]
        h = h[rng.integers(len(h))]
        n = len(h["y"])
        s = rng.integers(0, n - T)
        y = h["y"][s:s + T].astype(np.int64)
        if (y >= 0).mean() < 0.5 or ((y == 0) | (y == 1)).mean() < 0.3:
            continue
        X, M, touch = TD.window(h, s, T)
        ev, evf = widen(h["ev"][s:s + T], a.widen).astype(np.int64), widen(h["evf"][s:s + T], a.widen).astype(np.int64)
        ball = h["ball"][s:s + T].copy()
        if rng.random() < 0.5:
            X = TD.swap_teams(X); ball = -ball
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
            ev = np.where(ev == 1, 2, np.where(ev == 2, 1, 0)); evf = np.where(evf == 1, 2, np.where(evf == 2, 1, 0))
        if rng.random() < 0.5:
            X = TD.mirror(X); ball[:, 1] *= -1
        if a.noise:
            X[..., :2] += (rng.normal(0, a.noise, X[..., :2].shape) / np.array([52.5, 34.0])).astype(np.float32)
        M = M & ~(rng.random(M.shape[1]) < rng.uniform(0, a.dropout))[None]  # whole tracks hidden
        X[~M] = 0
        for k, v in zip(out, (X, M, y, ev, evf, ball, touch)):
            out[k].append(v)
    return {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}


def losses(o, b, a):
    y, X, M = b["y"], b["X"], b["M"]
    g = y >= 0
    w = torch.tensor([1.0, a.pos, a.pos])
    L = {"ev": F.cross_entropy(o["ev"][g], b["ev"][g], weight=w), "evf": F.cross_entropy(o["evf"][g], b["evf"][g], weight=w),
         "cls": F.cross_entropy(o["cls"][g], y[g])}
    scale = torch.tensor([52.5, 34.0])
    bb = b["ball"] * scale
    seen = torch.isfinite(bb[..., 0]) & g & (y < 2) & M.any(-1)
    d = ((X[..., :2] * scale - torch.nan_to_num(bb).unsqueeze(2)) ** 2).sum(-1).sqrt().masked_fill(~M, 1e4)
    tgt = torch.softmax(-d / 2.0, -1)
    L["ptr"] = -(tgt[seen] * F.log_softmax(o["ptr"][seen], -1)).sum(-1).mean() if seen.any() else o["ptr"].sum() * 0
    tt = torch.zeros_like(o["touch"])
    bi, ti = torch.nonzero(b["touch"] >= 0, as_tuple=True)
    for k in (-1, 0, 1):
        t2 = (ti + k).clamp(0, tt.shape[1] - 1)
        tt[bi, t2, b["touch"][bi, ti]] = 1
    tm = M & seen.unsqueeze(-1)
    L["touch"] = F.binary_cross_entropy_with_logits(o["touch"][tm], tt[tm], pos_weight=torch.tensor(5.0)) if tm.any() else o["touch"].sum() * 0
    return L


@torch.no_grad()
def predict_half(net, h, chunk=1500, overlap=100):
    """ev, evf, cls (T, 3) for a half, two symmetric views averaged; in overlapping chunks."""
    net.eval()
    n = len(h["X"])
    outs = {k: np.zeros((n, 3), np.float32) for k in ("ev", "evf", "cls")}
    s = 0
    while s < n:
        e = min(n, s + chunk)
        s0, e0 = max(0, s - overlap), min(n, e + overlap)
        X, M, _ = TD.window(dict(X=h["X"], M=h["M"]), s0, e0 - s0)
        acc = {k: 0 for k in outs}
        for sw, mi in ((0, 0), (1, 1)):
            if True:
                Xv = X
                if sw:
                    Xv = TD.swap_teams(Xv)
                if mi:
                    Xv = TD.mirror(Xv)
                o = net(torch.from_numpy(Xv)[None], torch.from_numpy(M)[None])
                for k in outs:
                    p = F.softmax(o[k][0], -1).numpy()
                    if sw:
                        p = p[:, [0, 2, 1]] if k != "cls" else p[:, [1, 0, 2]]
                    acc[k] = acc[k] + p / 2
        for k in outs:
            outs[k][s:e] = acc[k][s - s0:s - s0 + (e - s)]
        s = e
    return outs


def ap(net, ms, key="ev"):
    """Average precision of 'a turnover within 0.24 s of this frame' over live frames (direction ignored)."""
    sc, tg = [], []
    for m in ms:
        for h in m["halves"]:
            o = predict_half(net, h)
            g = (h["y"] == 0) | (h["y"] == 1)
            sc.append(1 - o[key][g, 0]); tg.append(widen(h[key], 3)[g] > 0)
    if not sc:
        return float("nan")
    sc, tg = np.concatenate(sc), np.concatenate(tg)
    hit = tg[np.argsort(-sc)]
    prec = np.cumsum(hit) / (np.arange(len(hit)) + 1)
    return float((prec * hit).sum() / max(hit.sum(), 1))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--fold", required=True)
    ap_.add_argument("--steps", type=int, default=1200)
    ap_.add_argument("--batch", type=int, default=16)
    ap_.add_argument("--window", type=int, default=125)
    ap_.add_argument("--lr", type=float, default=1e-3)
    ap_.add_argument("--noise", type=float, default=0.1, help="position jitter, metres")
    ap_.add_argument("--dropout", type=float, default=0.15)
    ap_.add_argument("--widen", type=int, default=2, help="frames either side of a turnover that count as it")
    ap_.add_argument("--pos", type=float, default=3.0, help="loss weight of the turnover classes")
    ap_.add_argument("--ptr", type=float, default=0.5)
    ap_.add_argument("--touch", type=float, default=0.5)
    ap_.add_argument("--cls", type=float, default=0.3)
    ap_.add_argument("--threads", type=int, default=1)
    ap_.add_argument("--tag", default="t1")
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    matches = TD.load_all()
    train, val, test = split(matches, a.fold) if a.fold != "none" else (matches, [], [])
    net = TurnNet(nfeat=TD.NFEAT)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(ROOT, "data", "cache", "turnover", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "ckpt.pt")
    start, run, t0 = 0, [], time.time()
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"]); start = c["step"]
        print(f"resumed at step {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        net.train()
        b = batch(train, rng, a.batch, a.window, a)
        L = losses(net(b["X"], b["M"]), b, a)
        loss = L["ev"] + L["evf"] + a.cls * L["cls"] + a.ptr * L["ptr"] + a.touch * L["touch"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append([L[k].item() for k in ("ev", "evf", "cls", "ptr", "touch")])
        if step % 100 == 0 or step == a.steps:
            r = np.mean(run, 0); run = []
            msg = f"{step:5d} ev {r[0]:.4f} evf {r[1]:.4f} cls {r[2]:.3f} ptr {r[3]:.3f} touch {r[4]:.3f}  {time.time() - t0:.0f}s"
            if step % 400 == 0 or step == a.steps:
                msg += f"  AP val {ap(net, val[:1]):.3f} test {ap(net, test[:1]):.3f}"
            print(msg, flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), step=step), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    for kind, ms in (("val", val), ("test", test)):
        for m in ms:
            res = {}
            for i, h in enumerate(m["halves"]):
                o = predict_half(net, h)
                res.update({f"ev{i}": o["ev"].astype(np.float16), f"evf{i}": o["evf"].astype(np.float16), f"cls{i}": o["cls"].astype(np.float16)})
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind, **res)
    dst = f"/mnt/project-files/turnover/{a.tag}/{a.fold}/"
    if os.path.isdir("/mnt/project-files/"):
        import glob, shutil
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(out, "*.npz")) + [os.path.join(out, "net.pt")]:
            shutil.copy(f, dst)


if __name__ == "__main__":
    main()
