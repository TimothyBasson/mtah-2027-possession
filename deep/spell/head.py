"""Train a head on a fold net's frozen features with a per-frame or a whole-sequence loss.

    python deep/spell/head.py --fold idsse --name crf --loss crf [--arch gru] [--labels control]

The fold net (deep/train.py) stays as it is; only what reads its 128-d state is trained, on the
fold's training matches, so each variant costs minutes. --loss ce is the per-frame loss the net
was trained with, decoded the shipped way (the control for "a new head" and "other labels");
--loss crf trains through the minimum-duration chain in crf.py, which is also the decode.
--margin and --spell charge wrong frames in the loss, per possession rather than per frame.
--risk N then spends N steps on the score itself: paths are drawn from the chain, each scored
with spellF1 against the labels, and likelier paths are made of the better-scoring ones
(minimum-risk training; Shen et al. 2016).

Prints the held-out provider's spellF1 on the labels as they are and on the folded ones, and
saves the predicted states to data/cache/spell/<tag>/<fold>/heads/<name>.npz.
"""
import argparse
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
from crf import MinDurCRF  # noqa: E402
from decode import absorb, viterbi  # noqa: E402
from train import spell_weights  # noqa: E402


class Head(nn.Module):
    def __init__(self, arch, d=128, width=96, drop=0.2):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.gru = nn.GRU(d, width, batch_first=True, bidirectional=True) if arch == "gru" else None
        self.out = nn.Linear(d + (2 * width if self.gru is not None else 0), 3)

    def forward(self, h):
        h = self.drop(h)
        if self.gru is not None:
            h = torch.cat([h, self.gru(h)[0]], -1)
        return self.out(h)


def project(y, mins):
    """Labels with runs under the chain's minimum joined to a neighbour: the nearest thing to the
    labels that the chain can say."""
    return absorb(y, {0: mins[0], 1: mins[1], 2: mins[2], -1: 0}).astype(np.int64)


def prepare(ms, a):
    for m in ms:
        for h in m["halves"]:
            y = h["yc"] if a.labels == "control" else h["y"]
            h["t"] = project(y, a.mins) if a.loss == "crf" else y
            h["w"] = spell_weights(h["t"], a.spell) if a.spell else np.ones(len(y), np.float32)


def batch(train, rng, B, T):
    provs = sorted({m["provider"] for m in train})
    H, Y, W = [], [], []
    while len(H) < B:
        prov = provs[rng.integers(len(provs))]
        ms = [m for m in train if m["provider"] == prov]
        h = ms[rng.integers(len(ms))]["halves"][rng.integers(2)]
        s = rng.integers(0, len(h["t"]) - T)
        y = h["t"][s:s + T]
        if (y >= 0).mean() < 0.3:
            continue
        v = C.VIEWS[rng.integers(4)]
        H.append(h["h" + v][s:s + T].astype(np.float32))
        Y.append(C.SWAP[y] * (y >= 0) - (y < 0) if v[0] == "1" else y)
        W.append(h["w"][s:s + T])
    return torch.from_numpy(np.stack(H)), torch.from_numpy(np.stack(Y)), torch.from_numpy(np.stack(W))


def window_f1(y, p):
    """spellF1 of one window of 5 fps codes (0 H, 1 A, 2 D, -1 ungraded)."""
    gt = np.where(y < 0, C.pm.X, np.array([C.pm.H, C.pm.A, C.pm.D])[y])
    pr = np.array([C.pm.H, C.pm.A, C.pm.D])[p]
    runs = C.pm.valid_runs(gt)
    return C.pm.spell_scores(gt, pr, runs)["mean_f1"]


@torch.no_grad()
def emissions(head, m):
    """Per half, (T, 3) class scores averaged over the four views."""
    head.eval()
    out = []
    for h in m["halves"]:
        per = {v: head(torch.from_numpy(h["h" + v].astype(np.float32))[None])[0] for v in C.VIEWS}
        if head.probs:
            per = {v: F.softmax(e, -1) for v, e in per.items()}
        out.append(C.tta({v: e.numpy() for v, e in per.items()}))
    return out


def decode_frames(P, cost=4, min_team=1.0, min_dead=1.0):
    """The shipped decode at 5 fps: one switching cost on log-probabilities at 25 fps weight,
    then short runs absorbed."""
    path = viterbi(5 * np.log(np.clip(P, 1e-6, 1)), cost)
    return absorb(path, {0: min_team * 5, 1: min_team * 5, 2: min_dead * 5})


def decode_chain(crf, Es):
    T = max(len(e) for e in Es)
    E = torch.zeros(len(Es), T, 3)
    for i, e in enumerate(Es):
        E[i, :len(e)] = torch.from_numpy(e)
    paths = crf.viterbi(E, torch.tensor([len(e) for e in Es]))
    return [paths[i, :len(e)].numpy() for i, e in enumerate(Es)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--arch", default="lin", choices=("lin", "gru"))
    ap.add_argument("--loss", default="crf", choices=("ce", "crf"))
    ap.add_argument("--labels", default="control", choices=("raw", "control"))
    ap.add_argument("--mins", default="10,10,5", help="shortest H, A and D run of the chain, in 0.2 s steps")
    ap.add_argument("--margin", type=float, default=0.0, help="what a wrong frame is charged in the loss")
    ap.add_argument("--spell", type=float, default=0.0, help="power of the per-possession weight")
    ap.add_argument("--scales", default="1,2,3,5,8,12", help="evidence scales tried at the decode")
    ap.add_argument("--risk", type=int, default=0, help="steps of training on sampled paths' spellF1")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.mins = tuple(int(v) for v in a.mins.split(","))
    a.scales = tuple(float(v) for v in a.scales.split(","))
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train, val, test = C.load_fold(a.fold, a.tag)
    prepare(train, a)
    head = Head(a.arch)
    head.probs = a.loss == "ce"
    crf = MinDurCRF(a.mins) if a.loss == "crf" else None
    params = list(head.parameters()) + (list(crf.parameters()) if crf else [])
    total = a.steps + a.risk
    opt = torch.optim.AdamW(params, a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.1)
    t0, run = time.time(), []
    for step in range(1, total + 1):
        head.train()
        Hb, y, w = batch(train, rng, a.batch, a.window)
        E = head(Hb)
        g = y >= 0
        if a.loss == "ce":
            loss = (F.cross_entropy(E[g], y[g], reduction="none") * w[g]).sum() / w[g].sum()
        elif step <= a.steps:
            loss = crf.nll(E, y, a.margin * w if a.margin else None).sum() / g.sum()
        else:  # the score itself
            S = crf.sample(E, a.samples)
            paths = crf.cls[S].numpy()
            r = torch.tensor([[window_f1(yy, p) for p in ps] for yy, ps in zip(y.numpy(), paths)], dtype=torch.float32)
            logp = crf.path_score(E, S) - crf.log_z(crf.expand(E))[:, None]
            adv = r - r.mean(1, keepdim=True)
            loss = -(adv * logp).mean() / 10 + 0.1 * crf.nll(E, y).sum() / g.sum()
            run.append(r.mean().item())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step <= a.steps:
            run.append(loss.item())
        if step % 200 == 0 or step == total:
            print(f"  {a.fold} {a.name} {step:5d} {'risk f1' if step > a.steps else 'loss'} {np.mean(run):.4f}  {time.time() - t0:.0f}s",
                  file=sys.stderr, flush=True)
            run = []
    # predict and score: validation matches (training providers, unseen) and the held-out provider
    # A chain fitted by likelihood trusts its evidence less than the score wants (frames repeat each
    # other, and its best path then joins possessions), so the evidence is scaled before the decode
    # and the scale is picked on the validation matches.
    ms = [(kind, m) for kind, group in (("val", val), ("test", test)) for m in group]
    Es = [emissions(head, m) for _, m in ms]
    by_scale = {}
    for k in (a.scales if crf else (1.0,)):
        res = {}
        flat = decode_chain(crf, [k * e for E2 in Es for e in E2]) if crf else [decode_frames(P) for E2 in Es for P in E2]
        allp = []
        for (kind, m), E2 in zip(ms, Es):
            paths, flat = flat[:len(E2)], flat[len(E2):]
            allp.append(paths)
            for view in ("raw", "control"):
                res.setdefault((kind, m["provider"], view), []).append(C.spell_f1(m, paths, view))
        vals = {view: np.mean([np.mean(v) for (kd, p, vw), v in res.items() if kd == "val" and vw == view]) for view in ("raw", "control")}
        tests = {view: np.mean(res[("test", a.fold, view)]) for view in ("raw", "control")}
        by_scale[k] = (vals, tests, allp)
        print(f"  {a.fold} {a.name} scale {k:g} test raw {tests['raw']:.4f} control {tests['control']:.4f} "
              f"val raw {vals['raw']:.4f} control {vals['control']:.4f}", file=sys.stderr, flush=True)
    pick = max(by_scale, key=lambda k: by_scale[k][0][a.labels])
    vals, tests, allp = by_scale[pick]
    save = {}
    for (kind, m), E2, paths in zip(ms, Es, allp):
        for i, (e, p) in enumerate(zip(E2, paths)):
            save[f"{m['name']}__e{i}"], save[f"{m['name']}__s{i}"] = e.astype(np.float32), p.astype(np.int8)
    out = os.path.join(C.ROOT, "data", "cache", "spell", a.tag, a.fold, "heads")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"{a.name}_s{a.seed}.npz"), **save)
    torch.save(dict(head=head.state_dict(), crf=crf.state_dict() if crf else None, args=vars(a), scale=pick),
               os.path.join(out, f"{a.name}_s{a.seed}.pt"))
    extra = ""
    if crf:
        extra = f" scale {pick:g} costs " + " ".join(f"{x:.1f}" for x in crf.costs().detach().reshape(-1)[[1, 2, 3, 5, 6, 7]])
    print(f"RESULT {a.name} seed {a.seed} {a.fold} test raw {tests['raw']:.4f} control {tests['control']:.4f} "
          f"val raw {vals['raw']:.4f} control {vals['control']:.4f}{extra}", flush=True)


if __name__ == "__main__":
    main()
