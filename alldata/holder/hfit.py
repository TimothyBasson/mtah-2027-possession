"""The all-data model with a per-player "who has the ball" branch; one provider held out.

    python alldata/holder/hfit.py --fold idsse --tag k1 --share skillcorner=0.25 [--pseudo k1e]

alldata/learned/fit.py sees the players only through summaries of each team's three nearest to
the ball. Here the five nearest players of each team are tokens of their own, with inputs
relative to the ball (offset, distance and its rank, closing speed, where the ball's path passes
the player and when, and the player's own approach to the ball over the second either side). The
tokens of a frame attend to each other and each is scored: the scores are a pointer over the
players, "who has the ball". The pointer is taught two ways. On live frames its mass on the
labelled team's players is pushed up (needs no holder label), and where the ball as recorded
lies within HOLD_R of a player of the labelled team that player is the target, while the network
is shown the BAR-ATH-like ball: the sharp ball teaches how to read the smoothed one. Each team's
pointer-weighted player state and pointer mass join the frame rows in the trunk of fit.py, which
is otherwise unchanged (same rows, heads, loss, sampling, outputs), so runs of the two score
side by side with alldata/learned/score.py.

A first build that carried every tracked player along his own path in time (dilated
convolutions per player) cost 8 s a step on one CPU against 0.7 s for fit.py, twelve times the
training time, and was dropped for this one (1.5 s a step) before it was scored.
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned")]
import feats  # noqa: E402
import fit as B  # noqa: E402

PLAYERS = os.path.join(B.CACHE, "players")
KMAX = 26      # player columns per training window
HOLD_R = 2.5   # m
NPF = 22


def player_inputs(X, P, team, ix):
    """(B, T, K, NPF) ball-relative inputs and the (B, T, K) tracked mask.
    X (B, T, F) feature rows, P (B, T, K, 4) with NaN where untracked, team (B, K) 0 home 1 away."""
    ok = torch.isfinite(P[..., 0])
    P = torch.nan_to_num(P)
    b = torch.stack([X[..., ix[0]] * 52.5, X[..., ix[1]] * 34.0], -1)[:, :, None]
    bv = torch.stack([X[..., ix[2]], X[..., ix[3]]], -1)[:, :, None] * 10
    seen, since = X[..., ix[4]][:, :, None], X[..., ix[5]][:, :, None]
    p, pv = P[..., :2], P[..., 2:]
    r = b - p
    d = r.norm(dim=-1)
    u = r / d.clamp(min=0.3)[..., None]
    rv = bv - pv
    closing = -(rv * u).sum(-1)              # how fast the gap shrinks
    sp = bv.norm(dim=-1)
    bu = bv / sp.clamp(min=0.5)[..., None]
    along = -(r * bu).sum(-1)                # the player's place along the ball's path, metres ahead of it
    across = (r[..., 0] * bu[..., 1] - r[..., 1] * bu[..., 0]).abs()   # and how far beside it
    eta = (along / sp.clamp(min=1.0)).clamp(-3, 5)
    moving = (sp > 1.0).float()
    dm = torch.where(ok, d, torch.full_like(d, 1e4))
    lt = dm[..., None, :] < dm[..., :, None]                       # [.., i, j]: j nearer than i
    same = team[:, None, :, None] == team[:, None, None, :]
    rank, rank_own = lt.sum(-1).float(), (lt & same).sum(-1).float()
    side = (1.0 - 2.0 * team.float())[:, None, :].expand_as(d)     # +1 home, -1 away
    one = torch.ones_like(d)
    f = torch.stack([
        (r[..., 0] / 10).clamp(-3, 3), (r[..., 1] / 10).clamp(-3, 3), torch.log1p(d.clamp(max=60)) / 2,
        torch.exp(-d / 1.5), torch.exp(-d / 4.0), pv[..., 0] / 5, pv[..., 1] / 5,
        (rv[..., 0] / 10).clamp(-3, 3), (rv[..., 1] / 10).clamp(-3, 3), (closing / 5).clamp(-3, 3),
        (along / 10).clamp(-3, 3) * moving, torch.log1p(across.clamp(max=30)) / 2 * moving, eta / 3 * moving,
        moving * one, seen * one, since * one, side, p[..., 0] / 52.5, p[..., 1] / 34.0,
        rank.clamp(max=10) / 5, rank_own.clamp(max=5) / 3, (rank == 0).float()], -1)
    return f * ok[..., None], ok


OFFS = (-5, -2, -1, 1, 2, 5)   # rows: 0.2, 0.4 and 1 s either side
NTOK = 5                       # players per team that become tokens: the nearest to the ball
NIN = NPF + 2 * len(OFFS) + 1


def path_inputs(f, ok, d):
    """Each player's own approach to the ball over the second either side: his distance to it
    at each offset against now, and the nearest he comes. A player untracked at the offset
    counts as where he is now."""
    dm = torch.where(ok, d, torch.full_like(d, 60.0)).clamp(max=60)
    cols, lo = [], dm
    for o in OFFS:
        sh = torch.roll(dm, -o, 1)
        oks = torch.roll(ok, -o, 1)
        if o > 0:
            sh[:, -o:], oks[:, -o:] = dm[:, -o:], False
        else:
            sh[:, :-o], oks[:, :-o] = dm[:, :-o], False
        sh = torch.where(oks, sh, dm)
        lo = torch.minimum(lo, sh)
        cols += [((sh - dm) / 5).clamp(-3, 3), torch.exp(-sh / 1.5)]
    return torch.cat([f, torch.stack(cols + [torch.exp(-lo / 1.5)], -1) * ok[..., None]], -1)


class HolderNet(nn.Module):
    def __init__(self, nf, names, d=96, dp=48, layers=1, dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32), drop=0.1):
        super().__init__()
        self.ix = [names.index(n) for n in ("bx", "by", "vx", "vy", "seen", "since")]
        self.penc = nn.Sequential(nn.Linear(NIN, dp), nn.GELU(), nn.Linear(dp, dp))
        self.ctx = nn.Linear(nf, dp)
        self.att = nn.ModuleList([nn.MultiheadAttention(dp, 4, batch_first=True) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dp) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dp) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(dp, 2 * dp), nn.GELU(), nn.Linear(2 * dp, dp)) for _ in range(layers)])
        self.point = nn.Linear(dp, 1)
        self.fuse = nn.Sequential(nn.Linear(2 * dp + 2, d), nn.GELU(), nn.Linear(d, d))
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.tcn = nn.Sequential(*[B.TCNBlock(d, k, drop) for k in dilations])
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(d, 6)
        self.edge = nn.Linear(d, 1)

    def tokens(self, X, P, team):
        """The NTOK nearest players of each team as tokens (home's first): inputs (B, T, 2 NTOK,
        NIN), tracked mask, and the player column behind each token."""
        f, ok = player_inputs(X, P, team, self.ix)
        d = torch.expm1(f[..., 2] * 2)
        f = path_inputs(f, ok, torch.where(ok, d, torch.full_like(d, 1e4)))
        Bn, T, K, _ = f.shape
        tm = team[:, None, :].expand(Bn, T, K)
        sel = []
        for t in (0, 1):
            key = torch.where(ok & (tm == t), d, torch.full_like(d, 1e4) + (tm != t) * 1e4)
            sel.append(key.topk(min(NTOK, K), -1, largest=False).indices)
        sel = torch.cat(sel, -1)
        tok = f.gather(2, sel[..., None].expand(-1, -1, -1, f.shape[-1]))
        n = sel.shape[-1] // 2
        side = torch.arange(2 * n, device=sel.device)[None, None, :] >= n
        valid = ok.gather(2, sel) & (tm.gather(2, sel) == side)
        return tok, valid, sel, side.expand_as(valid)

    def forward(self, X, P, team):
        tok, ok, sel, side = self.tokens(X, P, team)
        Bn, T, N, _ = tok.shape
        e = ((self.penc(tok) + self.ctx(X)[:, :, None]) * ok[..., None]).reshape(Bn * T, N, -1)
        pad = ~ok.reshape(Bn * T, N)
        pad = pad & ~pad.all(1, keepdim=True)      # a frame with nobody tracked attends to zeros
        for att, n1, n2, ff in zip(self.att, self.n1, self.n2, self.ff):
            z = n1(e)
            e = e + att(z, z, z, key_padding_mask=pad, need_weights=False)[0]
            e = e + ff(n2(e))
        e = e.reshape(Bn, T, N, -1) * ok[..., None]
        s = self.point(e).squeeze(-1).masked_fill(~ok, -1e4)
        lp = F.log_softmax(s, -1)                  # the pointer over the tokens
        pools, mass = [], []
        for t in (False, True):
            st = s.masked_fill(side != t, -1e4)
            pools.append((F.softmax(st, -1) * (st > -1e3))[..., None].mul(e).sum(2))
            mass.append(torch.logsumexp(lp.masked_fill((side != t) | ~ok, -1e4), -1).clamp(min=-8))
        lm = torch.stack(mass, -1)                 # (B, T, 2) log of the pointer's mass on each team
        h = self.inp(X) + self.fuse(torch.cat(pools + [lm / 4], -1))
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.gru(h)[0]
        return self.cls(h), self.edge(h).squeeze(-1), lp, lm, sel


def attach_players(ms):
    """Each half gets its players' paths, and the holder read off the ball as recorded: the
    nearest player of the labelled team, where he is within HOLD_R of it (else -1)."""
    for m in ms:
        z = np.load(os.path.join(PLAYERS, m["name"] + ".npz"))
        bx, by, sn = (m["names"].index(n) for n in ("bx", "by", "seen"))
        for i, h in enumerate(m["halves"]):
            P, team = z[f"P_{i}"], z[f"team_{i}"].astype(np.int64)
            assert len(P) == len(h["y"]), (m["name"], len(P), len(h["y"]))
            X0 = h["F"][0]
            b = np.stack([X0[:, bx] * 52.5, X0[:, by] * 34.0], 1)
            d = np.linalg.norm(np.nan_to_num(P[:, :, :2].astype(np.float32), nan=1e4) - b[:, None], axis=-1)
            d[team[None, :] != h["y"][:, None]] = 1e4
            j = d.argmin(1)
            good = (h["y"] >= 0) & (h["y"] <= 1) & (X0[:, sn] > 0.5) & (d[np.arange(len(j)), j] < HOLD_R)
            h["P"], h["team"], h["hold"] = P, team, np.where(good, j, -1)
    return ms


def pseudo(test, tag, fold, conf, pv):
    out = []
    for m in test:
        q = B.pseudo_labels(m, tag, fold, conf, pv)
        for h, g in zip(m["halves"], q["halves"]):
            g["P"], g["team"], g["hold"] = h["P"], h["team"], np.full(len(g["y"]), -1)
        out.append(q)
    return out


def batch(train, rng, Bn, T, a, sym):
    """fit.batch's draw (providers by --share, simulated ball versions, the two symmetries), with
    the players of each window."""
    perm, sw, mi = sym
    provs = sorted({m["provider"] for m in train})
    fixed = {k: v for k, v in a.share.items() if k in provs}
    rest = (1 - sum(fixed.values())) / max(len(provs) - len(fixed), 1)
    pw = np.array([fixed.get(k, rest) for k in provs])
    o = dict(X=[], y=[], e=[], y2=[], P=[], team=[], hold=[])
    while len(o["X"]) < Bn:
        prov = provs[rng.choice(len(provs), p=pw / pw.sum())]
        ms = [m for m in train if m["provider"] == prov]
        h = ms[rng.integers(len(ms))]["halves"]
        h = h[rng.integers(len(h))]
        v = rng.integers(1, min(len(h["F"]), 4)) if rng.random() < a.sim and len(h["F"]) > 1 else 0
        s = rng.integers(0, max(1, len(h["y"]) - T))
        X, y, y2 = h["F"][v][s:s + T], h["y"][s:s + T], h["yf"][s:s + T]
        if len(X) < T or (y >= 0).mean() < 0.3:
            continue
        P = h["P"][s:s + T].astype(np.float32)
        n = np.isfinite(P[:, :, 0]).sum(0)
        cols = np.sort(np.argsort(-n, kind="stable")[:KMAX])
        cols = cols[n[cols] > 0]
        P, team = P[:, cols], h["team"][cols]
        back = np.full(h["P"].shape[1] + 1, -1)
        back[cols] = np.arange(len(cols))
        hold = back[h["hold"][s:s + T]]
        if rng.random() < 0.5:
            X, y, y2, P, team = X[:, perm] * sw, B.swap_y(y), B.swap_y(y2), -P, 1 - team
        if rng.random() < 0.5:
            X, P = X * mi, P * np.array([1, -1, 1, -1], np.float32)
        k = KMAX - len(cols)
        o["X"].append(X.astype(np.float32)); o["y"].append(y); o["e"].append(B.edges(y)); o["y2"].append(y2)
        o["P"].append(np.pad(P, ((0, 0), (0, k), (0, 0)), constant_values=np.nan))
        o["team"].append(np.pad(team, (0, k))); o["hold"].append(hold)
    return {k: torch.from_numpy(np.stack(v)) for k, v in o.items()}


@torch.no_grad()
def predict(net, X, P, team, sym):
    """(T, 6) probabilities (labels as built, folded), averaged over the four symmetric views."""
    perm, sw, mi = sym
    net.eval()
    P = P.astype(np.float32)
    out = 0
    for s in (0, 1):
        for m in (0, 1):
            Z, Q, tm = X, P, team
            if s:
                Z, Q, tm = Z[:, perm] * sw, -Q, 1 - tm
            if m:
                Z, Q = Z * mi, Q * np.array([1, -1, 1, -1], np.float32)
            t = lambda v: torch.from_numpy(np.ascontiguousarray(v))[None]  # noqa: E731
            lg = net(t(Z.astype(np.float32)), t(Q), t(tm))[0][0]
            p = torch.cat([F.softmax(lg[:, :3], -1), F.softmax(lg[:, 3:], -1)], -1).numpy()
            out = out + (p[:, [1, 0, 2, 4, 3, 5]] if s else p)
    return out / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)   # idsse | metrica | skillcorner | none
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--dp", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sim", type=float, default=1.0)
    ap.add_argument("--edge", type=float, default=0.3)
    ap.add_argument("--mass", type=float, default=0.3, help="weight of the pointer's mass on the labelled team")
    ap.add_argument("--hold", type=float, default=0.3, help="weight of the holder read off the recorded ball")
    ap.add_argument("--share", nargs="*", default=[])
    ap.add_argument("--pv", type=int, default=1)
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.share = {k: float(v) for k, v in (x.split("=") for x in a.share)}
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ms = attach_players(B.load_matches(drop_cols=["tim"]))
    train, val, test = B.split(ms, a.fold)
    if a.fold == "none":
        train, val = train + val, []
        test = attach_players(B.load_matches(drop_cols=["tim"], folder=os.path.join(B.CACHE, "bar")))
    names = ms[0]["names"]
    if a.pseudo:
        train = train + pseudo(test, a.pseudo, a.fold, a.conf, a.pv)
    sym = feats.symmetry(names)
    net = HolderNet(len(names), names, a.d, a.dp)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    out = os.path.join(B.CACHE, "runs", a.tag, a.fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for step in range(a.steps):
        net.train()
        b = batch(train, rng, a.batch, a.T, a, sym)
        y, y2 = b["y"], b["y2"]
        logits, edge, lp, lm, sel = net(b["X"], b["P"], b["team"])
        hit = sel == b["hold"][..., None]          # the holder's token, if he is among them
        slot = hit.float().argmax(-1)
        ok, ok2, live, held = y >= 0, y2 >= 0, (y >= 0) & (y <= 1), hit.any(-1)
        loss = (F.cross_entropy(logits[ok][:, :3], y[ok], label_smoothing=0.02)
                + F.cross_entropy(logits[ok2][:, 3:], y2[ok2], label_smoothing=0.02)
                + a.edge * F.binary_cross_entropy_with_logits(edge[ok], b["e"][ok]))
        lmass = -lm[live].gather(1, y[live][:, None]).mean() if live.any() else logits.sum() * 0
        lhold = -lp[held].gather(1, slot[held][:, None]).mean() if held.any() else logits.sum() * 0
        total = loss + a.mass * lmass + a.hold * lhold
        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            print(f"{a.tag} {a.fold} step {step} loss {loss.item():.3f} mass {lmass.item():.3f} hold {lhold.item():.3f} "
                  f"held {held.float().mean().item():.2f} {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(sd=net.state_dict(), names=names, d=a.d, dp=a.dp), os.path.join(out, "net.pt"))
    for kind, group in (("val", val), ("test", test)):
        for m in group:
            o = {}
            for i, h in enumerate(m["halves"]):
                for v, k in B.EVAL.items():
                    if v < len(h["F"]):
                        q = predict(net, h["F"][v], h["P"], h["team"], sym)
                        o[f"{k}{i}"], o[f"f{k}{i}"] = q[:, :3], q[:, 3:]
            np.savez_compressed(os.path.join(out, m["name"] + ".npz"), kind=kind, **o)
    print("done", a.tag, a.fold, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
