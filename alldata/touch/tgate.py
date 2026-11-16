"""A learned gate on forced possessions: which detected touches are a brief possession.

    python alldata/touch/tgate.py --tag pw1e --touch t1 [--v 1] [--thr 0.4 0.5 0.6]

A candidate is a detected touch by team K inside a possession the decode gives to team O, followed
within 5 s by a touch of O (tinsert.py). Forcing every candidate loses more than it gains; forcing
only those where the truth has K gains about +0.03 (the ceiling). The gate is a small boosted-tree
classifier on what is known about the candidate: the touch scores, how long until O touches again,
how many touches K has, what the net itself gives K there, who is nearest the ball, and where the
candidate sits in O's possession.

Held out by provider: the gate for a provider is trained on the candidates of the other two.
Every candidate, in training too, is built from predictions of nets and detectors that had the
match's provider held out, so it looks as an unseen provider's does (the same construction as
the players-only columns in alldata/learned/build.py). --save writes the gate trained on all
three providers.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import tuse  # noqa: E402
import tinsert  # noqa: E402
import tfeats  # noqa: E402

HOLD = 5.0
FCOLS = ["seen", "speed", "dv5", "dv10", "h0_d", "a0_d", "h1_d", "a1_d"]
VIEWS = ("built", "folded", "file")


def candidates(p, f, tm, sc, P, G, gnames, fps=25):
    """Rows of features and the (start, stop, team) of each candidate in decoded codes p.
    P (n, 3) the net's probabilities, G (n, len(FCOLS)) ball and nearest-player columns."""
    e = np.flatnonzero(np.diff(p) != 0) + 1
    rs, rt = np.r_[0, e], np.r_[e, len(p)]
    X, seg = [], []
    for j in range(len(f)):
        k = tm[j]
        o = tinsert.CODE[1 - k]
        a = f[j]
        if p[a] != o:
            continue
        if j and tm[j - 1] == k and p[f[j - 1]] == o and f[j] - f[j - 1] <= HOLD * fps and seg and seg[-1][1] > a:
            continue  # inside the previous candidate of the same team
        nx = np.flatnonzero((f > a) & (tm != k) & (f <= a + HOLD * fps))
        if not len(nx):
            continue
        b = f[nx[0]]
        if not (p[a:b] == o).all():
            continue
        own = (f >= a) & (f < b) & (tm == k)
        r = np.searchsorted(rt, a, "right")
        prev_o = np.flatnonzero((f < a) & (tm != k))
        gk = lambda fr, nm: G[fr, gnames.index(nm)]  # noqa: E731
        dk, do = ("h0_d", "a0_d") if k == 0 else ("a0_d", "h0_d")
        dk2 = "h1_d" if k == 0 else "a1_d"
        X.append([
            sc[j], sc[nx[0]], (b - a) / fps, own.sum(), sc[own].max(), sc[own].sum(),
            P[a:b, k].mean(), P[a:b, k].max(), P[a:b, 2].mean(), P[a, k], P[max(a - 12, 0), k], P[min(b + 12, len(P) - 1), k],
            (a - rs[r]) / fps, (rt[r] - b) / fps, (rt[r] - rs[r]) / fps,
            (a - f[prev_o[-1]]) / fps if len(prev_o) else 10.0, sc[prev_o[-1]] if len(prev_o) else 0.0,
            gk(a, dk), gk(a, do), gk(a, dk2), gk(a, "speed"), gk(a, "dv10"), gk(a, "dv5"),
            gk(b, dk), gk(b, do), gk(b, "speed"),
            gk(min(a + 12, len(G) - 1), dk), gk(min(a + 12, len(G) - 1), do),
            p[rs[r] - 1] == C.DEAD if rs[r] > 0 else 1, p[rt[r]] == C.DEAD if rt[r] < len(p) else 1,
        ])
        seg.append((a, b, k))
    return np.array(X, np.float32).reshape(len(seg), -1), seg


def match_data(args):
    fold, name, tag, touch, v = args
    raw, fo, spans = C.truths(name)
    n = len(raw)
    key = tuse.KEY[v]
    z = np.load(os.path.join(C.RUNS, tag, fold, name + ".npz"))
    nh = len(spans)
    pr = [z[f"{key}{i}"] for i in range(nh)]
    pf = [z[f"f{key}{i}"] for i in range(nh)]
    pb = [(x + y) / 2 for x, y in zip(pr, pf)]
    q = tuse.touch_probs(touch, fold, name, v, n, spans)
    f, tm, sc = tuse.events(q, 0.3)
    tz = np.load(os.path.join(tfeats.OUT, name + ".npz"))
    gn = [str(s) for s in tz["names"]]
    G = np.zeros((n, len(FCOLS)), np.float32)
    for i, (a, b) in enumerate(spans):
        G[a:b + 1] = tz[f"F{v}_{i}"][:, [gn.index(c) for c in FCOLS]]
    out = {}
    for view, probs, dec, truth in (("built", pr, C.RAW, raw), ("folded", pf, C.FOLDED, fo), ("file", pb, C.RAW, raw)):
        d = C.decode_ev(probs, spans, n, **dec)
        P = np.full((n, 3), 1 / 3)
        for i, (a, b) in enumerate(spans):
            P[a:b + 1] = C.decode.upsample(probs[i], b - a + 1)
        X, seg = candidates(d, f, tm, sc, P, G, FCOLS)
        # worth forcing: the truth has K for at least half of the forced frames
        y = np.array([(truth[a:b] == tinsert.CODE[k]).mean() >= 0.5 for a, b, k in seg])
        out[view] = dict(X=X, y=y, seg=seg, d=d)
    return (fold, name), out


def force(d, seg, keep):
    out = d.copy()
    for (a, b, k), ok in zip(seg, keep):
        if ok:
            out[a:b] = tinsert.CODE[k]
    return out


def fit_gate(X, y, w):
    from sklearn.ensemble import HistGradientBoostingClassifier
    g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, random_state=0)
    return g.fit(X, y, sample_weight=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--v", type=int, default=1)
    ap.add_argument("--thr", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    ms = tuse.matches(a.tag, "test")
    data = dict(C.pool_map(match_data, [(fo, n, a.tag, a.touch, a.v) for fo, n in ms], a.workers))
    print(f"== {a.tag} + gated forced possessions from touches {a.touch}, ball version {a.v}; IDSSE Metrica SkillCorner | mean")
    res = {v: {t: {} for t in ["base"] + a.thr} for v in VIEWS}
    for view in VIEWS:
        for held in C.PROVIDERS:
            tr = [(fo, n) for fo, n in ms if fo != held]
            X = np.concatenate([data[k][view]["X"] for k in tr])
            y = np.concatenate([data[k][view]["y"] for k in tr])
            # each training provider counts the same
            w = np.concatenate([np.full(len(data[k][view]["y"]), 1.0 / sum(len(data[j][view]["y"]) for j in tr if j[0] == k[0])) for k in tr])
            g = fit_gate(X, y, w * len(y) / w.sum())
            te = [(fo, n) for fo, n in ms if fo == held]
            pos = np.mean(np.concatenate([data[k][view]["y"] for k in te]))
            pr = np.concatenate([g.predict_proba(data[k][view]["X"])[:, 1] for k in te])
            yt = np.concatenate([data[k][view]["y"] for k in te])
            from sklearn.metrics import roc_auc_score
            print(f"  {view:7s} {held:12s} candidates {len(yt)}, true {pos:.2f}, gate AUC {roc_auc_score(yt, pr):.3f}, precision at 0.5 {yt[pr > 0.5].mean() if (pr > 0.5).any() else 0:.2f} taking {np.mean(pr > 0.5):.2f}")
            for k in te:
                D = data[k][view]
                p = g.predict_proba(D["X"])[:, 1] if len(D["X"]) else np.zeros(0)
                for t in ["base"] + a.thr:
                    res[view][t][k[1]] = D["d"] if t == "base" else force(D["d"], D["seg"], p > t)
    ev = lambda r, f_: {n: C.evaluate(n, r[n], f_[n]) for n in r}  # noqa: E731
    print("-- each head at its own decode, against its own labels")
    base = ev(res["built"]["base"], res["folded"]["base"])
    C.report("base", base)
    for t in a.thr:
        C.report(f"gate over {t}", ev(res["built"][t], res["folded"][t]), base)
    print("-- the file's decode (two heads averaged, 1 s minimum), one labelling against both truths")
    base = ev(res["file"]["base"], res["file"]["base"])
    C.report("base", base)
    for t in a.thr:
        C.report(f"gate over {t}", ev(res["file"][t], res["file"][t]), base)
    if a.save:
        X = np.concatenate([data[k]["file"]["X"] for k in ms])
        y = np.concatenate([data[k]["file"]["y"] for k in ms])
        w = np.concatenate([np.full(len(data[k]["file"]["y"]), 1.0 / sum(len(data[j]["file"]["y"]) for j in ms if j[0] == k[0])) for k in ms])
        with open(a.save, "wb") as fh:
            pickle.dump(fit_gate(X, y, w * len(y) / w.sum()), fh)
        print("saved", a.save)


if __name__ == "__main__":
    main()
