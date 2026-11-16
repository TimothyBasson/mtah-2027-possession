"""Train the segment corrector with one provider held out and score the corrected labels.

    python alldata/corrector/segfit.py [--segs segs_pw1e_v1] [--seeds 0 1] [--thr 0.5 0.6 0.7]

For each candidate segment (segs.py) a small boosted-tree model gives the probability that the
truth inside it is the other team; above the threshold the segment is relabelled. Prints
spellF1 per provider for the base decode, the oracle (every candidate that is truly the other
team relabelled), and each threshold; and the size of failure mode 1 before and after.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned")]
import score as S
from score import pm

CACHE = os.path.join(ROOT, "data", "cache", "alldata")
PROVS = ("idsse", "metrica", "skillcorner")
Hc, Ac, Dc, Xc = (int(v) for v in pm.label_codes(np.array(list("HADX"))))
CODE = np.array([Hc, Ac, Dc, Xc])


def choose(spans, score, thr):
    """Candidates scored above thr, best first, skipping any that overlaps one already kept."""
    keep = np.zeros(len(spans), bool)
    taken = []
    for i in np.argsort(-score):
        if score[i] <= thr:
            break
        s, e = spans[i]
        if all(e <= a or s >= b for a, b in taken):
            keep[i] = True
            taken.append((s, e))
    return keep


def apply(pred, spans, to, score, thr):
    q = pred.copy()
    for (s, e), y, f in zip(spans, to, choose(spans, score, thr)):
        if f:
            q[s:e] = y
    return q


def mode1(t, p, fps=25, max_len=5.0):
    """Failure mode 1: true possessions under max_len seconds, not matched at IoU 0.5, in which
    the prediction is mostly the other team. Returns their count and the spellF1 the prediction
    would reach if they were all made right (the oracle), so loss = oracle - score."""
    runs = pm.valid_runs(t)
    gts = pm.possession_spells(t, runs)
    prs = pm.possession_spells(p, runs)
    ov = pm.spell_overlaps(gts, prs)
    hit = pm.matched_gt_flags(ov, len(gts), 0.5)
    q = p.copy()
    cnt = 0
    for i, (a, b, k) in enumerate(gts):
        other = Ac if k == Hc else Hc
        if not hit[i] and (b - a) / fps < max_len and (p[a:b] == other).mean() > 0.5:
            q[a:b] = k
            cnt += 1
    return cnt, pm.evaluate(t, q, bootstrap_n=0)["spell_f1_mean"]


def score_one(args):
    name, tk, variants = args
    raw, fold, _ = S.truths(name)
    t = fold if tk == "f" else raw
    out = []
    for p in variants:
        f1 = pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"]
        c, orc = mode1(t, p)
        c10, orc10 = mode1(t, p, max_len=1e9)
        out.append((f1, c, orc - f1, c10, orc10 - f1))
    return out


def weights(tr, fold, sc_share, G):
    share = {p: 1.0 for p in PROVS if p != fold}
    if "skillcorner" in share and len(share) > 1:
        other = [p for p in share if p != "skillcorner"]
        share = {"skillcorner": sc_share, **{o: (1 - sc_share) / len(other) for o in other}}
    rows = {p: sum(len(G[n]["to"]) for n in tr if n.startswith(p)) for p in share}
    w = np.concatenate([np.full(len(G[n]["to"]), share[n.split("_")[0]] / max(rows[n.split("_")[0]], 1)) for n in tr])
    return w * len(w) / w.sum()


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1)


def target_reg(g, kind):
    u = g["u"] + g["uf"] if kind == "mix" else 2 * g["u" + kind]
    return np.clip(u, -3, 3)


def target(g, kind):
    """What the model learns. gain: relabelling the segment raises the score (as built, folded,
    or `mix`: on the two labellings together); team: the truth inside is the other team."""
    if kind == "team":
        return g["y"] > 0.5
    if kind == "mix":
        return g["u"] + g["uf"] > 0
    return g["u" + kind] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default="segs_pw1e_v1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--thr", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--sc-share", type=float, default=0.25)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--target", default="gain", choices=["gain", "team"], help="what the model learns: that relabelling the segment raises the score, or that the truth inside is the other team")
    ap.add_argument("--save")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--reg", action="store_true", help="learn the size of the score change, not its sign (thresholds are then in matched possessions)")
    a = ap.parse_args()
    z = np.load(os.path.join(CACHE, "corrector", a.segs + ".npz"))
    cols = [str(c) for c in z["names"]]
    keep = [i for i, c in enumerate(cols) if not any(c.startswith(d) for d in a.drop)]
    names = sorted({k.split("|")[0] for k in z.files if "|" in k})
    summary = []
    CONFIGS = [("", "", "", "head as built, scored as built"), ("f", "f", "f", "folded head, scored folded"),
               ("a", "", "mix", "shipped file (heads averaged), scored as built"), ("a", "f", "mix", "shipped file (heads averaged), scored folded")]
    done = {}
    for pre, tk, kind, title in CONFIGS:
        if a.target == "team":
            kind = "team"
        G = {n: {k: z[f"{n}|{pre}{k}"] for k in ("pred", "spans", "to", "rows")} for n in names}
        for n in names:
            G[n].update(y=z[f"{n}|{pre}y{tk}"], u=z[f"{n}|{pre}u"], uf=z[f"{n}|{pre}uf"])
        if (pre, kind) in done:
            prob = done[(pre, kind)]
        else:
            prob = {n: np.zeros(len(G[n]["to"])) for n in names}
            for fold in PROVS:
                tr = [n for n in names if not n.startswith(fold)]
                X = np.vstack([G[n]["rows"] for n in tr])[:, keep]
                y = np.concatenate([(target_reg if a.reg else target)(G[n], kind) for n in tr])
                w = weights(tr, fold, a.sc_share, G)
                for seed in a.seeds:
                    m = lgb.train(dict(PARAMS, seed=seed, **({"objective": "huber"} if a.reg else {})), lgb.Dataset(X, y, weight=w, feature_name=[cols[i] for i in keep]), a.rounds)
                    for n in names:
                        if n.startswith(fold) and len(G[n]["to"]):
                            prob[n] += m.predict(G[n]["rows"][:, keep]) / len(a.seeds)
                if fold == "idsse" and a.verbose:
                    g = m.feature_importance("gain")
                    print("  top:", ", ".join(f"{cols[keep[j]]} {g[j] / g.sum():.2f}" for j in np.argsort(-g)[:12]))
            done[(pre, kind)] = prob
        for n in names:
            G[n]["ut"] = z[f"{n}|{pre}u{tk}"]
        jobs = []
        for n in names:
            g = G[n]
            pred = CODE[g["pred"]]
            to = CODE[g["to"]]
            v = [pred, apply(pred, g["spans"], to, g["ut"], 0.0)] + [apply(pred, g["spans"], to, prob[n], th) for th in a.thr]
            jobs.append((n, tk, v))
        with ProcessPoolExecutor(4) as ex:
            R = np.array(list(ex.map(score_one, jobs)))  # (matches, variants, 5)
        pm_ = lambda k, vi: [R[[i for i, n in enumerate(names) if n.startswith(p)], vi, k].mean() for p in PROVS]
        ncand = [sum(len(G[n]["to"]) for n in names if n.startswith(p)) for p in PROVS]
        npos = [int(sum((G[n]["y"] > 0.5).sum() for n in names if n.startswith(p))) for p in PROVS]
        ngain = [int(sum((G[n]["ut"] > 0).sum() for n in names if n.startswith(p))) for p in PROVS]
        print(f"\n{title}: candidates {ncand}, truly the other team {npos}, relabel would raise the score {ngain}")
        print(f"  {'':12s} spellF1 idsse metrica skillc   mean | mode-1 loss under 5 s (count per match)        | mode-1 loss, any length")
        for vi, vn in enumerate(["base", "oracle"] + [f"thr {th}" for th in a.thr]):
            f, c, l, l2 = pm_(0, vi), pm_(1, vi), pm_(2, vi), pm_(4, vi)
            nfl = "" if vi < 2 else " flips/match " + " ".join(f"{np.mean([choose(G[n]['spans'], prob[n], a.thr[vi - 2]).sum() for n in names if n.startswith(p)]):5.1f}" for p in PROVS)
            print(f"  {vn:12s} {f[0]:.3f} {f[1]:.3f} {f[2]:.3f}  {np.mean(f):.3f} | {l[0]:.3f} {l[1]:.3f} {l[2]:.3f} mean {np.mean(l):.3f} ({c[0]:.0f} {c[1]:.0f} {c[2]:.0f}) | {np.mean(l2):.3f}{nfl}")
        if a.save:
            np.savez(a.save + pre + ".npz", **{n: prob[n] for n in names})
        summary.append((title, [(vn, pm_(0, vi), np.mean(pm_(2, vi)), np.mean(pm_(4, vi))) for vi, vn in enumerate(["base", "oracle"] + [f"thr {th}" for th in a.thr])]))
    return summary


if __name__ == "__main__":
    main()
