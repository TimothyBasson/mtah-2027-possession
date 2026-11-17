"""alldata/learned/fit.py unchanged, except that it also saves the change head's probability
(`e<k><half>`, the possession-change head every net already trains and nothing used).
PathCRF (Kim et al. 2026) learns its transition scores from the network; this is the cheap
version of that: the net's own change probability lowers the switch cost in the decode.

    python alldata/oss/fit_edge.py --fold idsse --tag ew1 --drop-cols tim --share skillcorner=0.25 --pseudo w1e
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "learned"))
from fit import *  # noqa: F401,F403,E402
from fit import CACHE, EVAL, Net, batch, feats, load_matches, pseudo_labels, split  # noqa: E402
import argparse, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


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
            lg, ed = net(torch.from_numpy(np.ascontiguousarray(Z, np.float32))[None])
            lg, ed = lg[0], ed[0]
            p = torch.cat([F.softmax(lg[:, :3], -1), F.softmax(lg[:, 3:], -1), torch.sigmoid(ed)[:, None]], -1).numpy()
            out = out + (p[:, [1, 0, 2, 4, 3, 5, 6]] if s else p)
    return out / 4  # (T, 7): labels as built, folded, change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)   # idsse | metrica | skillcorner | none
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sim", type=float, default=1.0)
    ap.add_argument("--spell", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--share", nargs="*", default=[], help="provider=share of training windows, e.g. skillcorner=0.25")
    ap.add_argument("--vers", nargs="*", type=int, default=[], help="simulated ball versions to train on (build.py: 1-3 parabolic, 4-6 low-pass, 7+ mix)")
    ap.add_argument("--pv", type=int, default=1, help="ball version the pseudo-labelled matches are shown in (1, 4 or 7)")
    ap.add_argument("--extra", default="", help="folder of extra input columns per match (alldata/touch/textra.py)")
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.share = {k: float(v) for k, v in (x.split("=") for x in a.share)}
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ms = load_matches(drop_cols=a.drop_cols, extra=a.extra)
    train, val, test = split(ms, a.fold)
    if a.fold == "none":  # every outside match trains; the unseen match is BAR-ATH (run.py --features)
        train, val = train + val, []
        test = load_matches(drop_cols=a.drop_cols, folder=os.path.join(CACHE, "bar"), extra=a.extra)
    names = ms[0]["names"]
    if a.pseudo:
        train = train + [pseudo_labels(m, a.pseudo, a.fold, a.conf, a.pv) for m in test]
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
                for v, k in EVAL.items():
                    if v < len(h["F"]):
                        q = predict(net, h["F"][v], sym)
                        o[f"{k}{i}"], o[f"f{k}{i}"], o[f"e{k}{i}"] = q[:, :3], q[:, 3:6], q[:, 6]
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind, **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
