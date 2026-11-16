"""v9's sequence model: deep/'s network, fed the ball as the decoder sees it, taught the control rule.

    python v9/train.py --fold idsse --tag sq      # and metrica, skillcorner: one provider held out
    python v9/train.py --fold none --tag sq --steps 2000

Two changes from deep/'s bt, each a switch so that either can be measured alone:

- --ballin: the ball is an input, per player (offset to the ball, nearness, the ball's velocity,
  ball-seen: deep/train.py with_ball). deep/ may not read the ball; hmm/ may, and v9 is a
  successor of hmm/. On the training matches the ball is hidden in runs (2 s mean, as
  external/score.py drop_ball) down to a share drawn per window, so the net cannot lean on a ball
  that BAR-ATH shows 59% of the time. On the matches it labels it gets exactly the ball
  external/score_control.py gives hmm_v8 (control_features.match_ball).
- --labels folded: the target is the label folded towards the challenge's control rule
  (external/control.py), the target hmm_v8's classifier has, instead of the provider's own.

The pointer and the three ball side heads of bt are kept, taught by the full ball of the training
matches. Writes data/cache/v9/<tag>/<fold>/<match>.npz, H/A/D probabilities at 5 fps.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), ROOT]
import data as D  # noqa: E402
import train as T  # noqa: E402
from control import control_labels  # noqa: E402
from control_features import match_ball  # noqa: E402
from net import PossessionNet  # noqa: E402

CODES = {"H": 0, "A": 1, "D": 2}


def prepare(matches, folded):
    """Per half: seen (the ball the scorer shows the decoder is there), and y folded if asked."""
    import pickle
    cache = os.path.join(ROOT, "data", "cache", "v9_extra.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            extra = pickle.load(f)
    else:
        extra = {}
        for m in matches:
            d = dict(np.load(D.match_path(m["name"])))
            lab = control_labels(d)
            code = np.full(len(lab), -1, np.int64)
            for c, i in CODES.items():
                code[lab == c] = i
            seen = np.isfinite(match_ball(m["name"], d)[:, 0])
            extra[m["name"]] = [(code[h["frames"]], seen[h["frames"]]) for h in m["halves"]]
        with open(cache, "wb") as f:
            pickle.dump(extra, f, protocol=4)
    for m in matches:
        for h, (yf, seen) in zip(m["halves"], extra[m["name"]]):
            h["seen"] = seen
            if folded:
                h["y"] = yf


def hide(n, rng, lo, hi):
    """Keep-mask of n steps: on/off runs, off 2 s on average, to a seen share drawn from lo..hi."""
    s = rng.uniform(lo, hi)
    off = 2 * 25 / D.STEP
    on = off * s / (1 - s)
    keep = np.ones(n, bool)
    i, state = 0, rng.random() < s
    while i < n:
        k = 1 + int(rng.exponential(on if state else off))
        keep[i:i + k] = state
        i, state = i + k, not state
    return keep


def batch(train, rng, B, Tn, a):
    provs = sorted({m["provider"] for m in train})
    Xs, Ms, ys, bs, es, ws, ds = [], [], [], [], [], [], []
    while len(Xs) < B:
        pi = rng.integers(len(provs))
        prov = provs[pi]
        ms = [m for m in train if m["provider"] == prov]
        m = ms[rng.integers(len(ms))]
        h = m["halves"][rng.integers(len(m["halves"]))]
        s = rng.integers(0, max(1, len(h["y"]) - Tn))
        X, M, y, ball = h["X"][s:s + Tn], h["M"][s:s + Tn], h["y"][s:s + Tn], h["ball"][s:s + Tn]
        if len(X) < Tn or (y >= 0).mean() < 0.3:
            continue
        e = T.edges(y)
        if rng.random() < 0.5:
            X, y, ball = D.swap_teams(X, y, ball)
        if rng.random() < 0.5:
            X, ball = D.mirror(X, ball)
        X, M = X.copy(), M.copy()
        if a.noise:
            X[..., :2] += rng.normal(0, a.noise / 40, X[..., :2].shape).astype(np.float32)
        M &= ~(rng.random(M.shape) < rng.uniform(0, a.dropout))
        X[~M] = 0
        if a.ballin:
            shown = ball.copy()
            # Metrica's ball is already missing in stretches: leave it as recorded half the time
            if not (prov == "metrica" and rng.random() < 0.5):
                shown[~hide(len(shown), rng, a.seen_lo, a.seen_hi)] = np.nan
            if a.bnoise:
                shown = shown + rng.normal(0, a.bnoise, shown.shape).astype(np.float32) / np.array([52.5, 34.0], np.float32)
            X = T.with_ball(X, shown)
        Xs.append(X); Ms.append(M); ys.append(y); bs.append(ball); es.append(e)
        ws.append(np.full(len(y), a.provw.get(prov, 1.0), np.float32)); ds.append(pi)
    t = lambda v: torch.from_numpy(np.stack(v))  # noqa: E731
    return t(Xs), t(Ms), t(ys), t(bs), t(es), t(ws), torch.tensor(ds)


@torch.no_grad()
def predict_half(net, h, ballin, tta=True):
    """(T, 3) probabilities over the four symmetric views. The ball shown is the scorer's."""
    net.eval()
    out = 0
    shown = np.where(h["seen"][:, None], h["ball"], np.nan).astype(np.float32)
    dummy = np.zeros(len(h["y"]), np.int64)
    views = [(s, m) for s in (0, 1) for m in (0, 1)] if tta else [(0, 0)]
    for s, m in views:
        X, b = h["X"], shown
        if s:
            X, _, b = D.swap_teams(X, dummy, b)
        if m:
            X, b = D.mirror(X, b)
        if ballin:
            X = T.with_ball(X, b)
        logits = net(torch.from_numpy(X)[None], torch.from_numpy(h["M"])[None])[0][0]
        p = F.softmax(logits, -1).numpy()
        out = out + (p[:, [1, 0, 2]] if s else p)
    return out / len(views)


def acc(net, ms, ballin):
    hit, n = 0, 0
    for m in ms:
        for h in m["halves"]:
            p = predict_half(net, h, ballin, tta=False)
            g = h["y"] >= 0
            hit += (p.argmax(1)[g] == h["y"][g]).sum(); n += g.sum()
    return hit / max(n, 1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.k * g, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--ballin", action="store_true")
    ap.add_argument("--labels", default="raw", choices=("raw", "folded"))
    ap.add_argument("--seen-lo", type=float, default=0.45)
    ap.add_argument("--seen-hi", type=float, default=0.85)
    ap.add_argument("--bnoise", type=float, default=0.0, help="ball position jitter, metres")
    ap.add_argument("--provw", default="")
    ap.add_argument("--adv", type=float, default=0.0,
                    help="domain-adversarial: weight of a which-provider head behind a gradient reversal")
    ap.add_argument("--ptr", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="sq")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.provw = {k: float(v) for k, v in (kv.split("=") for kv in a.provw.split(",") if kv)}
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    matches = [m for m in D.load_all() if m["provider"] in ("idsse", "metrica", "skillcorner")]
    prepare(matches, a.labels == "folded")
    train, val, test = T.split(matches, a.fold) if a.fold != "none" else (matches, [], [])
    net = PossessionNet(nfeat=D.NFEAT + (T.NBALL if a.ballin else 0), d=a.d, ptr=a.ptr > 0)
    dom = torch.nn.Sequential(torch.nn.Linear(a.d, 64), torch.nn.GELU(), torch.nn.Linear(64, 3)) if a.adv else None
    params = list(net.parameters()) + (list(dom.parameters()) if dom else [])
    opt = torch.optim.AdamW(params, a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(ROOT, "data", "cache", "v9", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    feats = {}
    if dom:  # the features the label head reads
        net.cls.register_forward_hook(lambda mod, inp, o: feats.__setitem__("h", inp[0]))
    t0, run, start = time.time(), [], 0
    ckpt = os.path.join(out, "ckpt.pt")
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        if dom:
            dom.load_state_dict(c["dom"])
        rng.bit_generator.state = c["rng"]; torch.set_rng_state(c["torch_rng"])
        start = c["step"]
        print(f"resumed at step {start}", flush=True)
    for step in range(start + 1, a.steps + 1):
        net.train()
        X, M, y, ball, e, w, prov = batch(train, rng, a.batch, a.window, a)
        logits, bhat, ehat, extra = net(X, M)
        g = y >= 0
        loss_c = (F.cross_entropy(logits[g], y[g], reduction="none") * w[g]).sum() / w[g].sum()
        loss_e = F.binary_cross_entropy_with_logits(ehat[g], e[g])
        loss = loss_c + 0.3 * loss_e
        if not a.ballin:  # where-is-the-ball head: pointless when the ball is an input
            bg = g & (y < 2) & torch.isfinite(ball[..., 0])
            if bg.any():
                loss = loss + 0.5 * F.smooth_l1_loss(bhat[bg], ball[bg], beta=0.1)
        lx = 0.0
        if a.ptr > 0:
            bl = T.ball_losses(extra, T.ball_targets(X[..., :D.NFEAT], M, ball, y))
            loss = loss + a.ptr * bl["ptr"] + 0.3 * bl["bvel"] + 0.3 * bl["touch"] + 0.2 * bl["free"]
            lx = bl["ptr"].item()
        ld = 0.0
        if dom:
            k = a.adv * min(1.0, step / (0.3 * a.steps))
            dl = F.cross_entropy(dom(GradReverse.apply(feats["h"], k))[g], prov[:, None].expand_as(y)[g])
            loss = loss + dl
            ld = dl.item()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run.append((loss_c.item(), loss_e.item(), lx, ld))
        if step % 100 == 0 or step == a.steps:
            r = np.mean(run, 0); run = []
            msg = f"{step:5d} ce {r[0]:.3f} edge {r[1]:.3f} ptr {r[2]:.3f} dom {r[3]:.3f}  {time.time() - t0:.0f}s"
            if step % 500 == 0 or step == a.steps:
                msg += f"  val acc {acc(net, val, a.ballin):.3f}  test acc {acc(net, test[:3], a.ballin):.3f}"
            print(msg, flush=True)
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            dom=dom.state_dict() if dom else None, rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), step=step), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    for kind, ms in (("val", val), ("test", test)):
        for m in ms:
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind,
                                **{f"p{i}": predict_half(net, h, a.ballin) for i, h in enumerate(m["halves"])})


if __name__ == "__main__":
    main()
