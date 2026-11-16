"""Where a saved run loses spellF1: oracles and misses by possession length.

    python alldata/learned/diag.py --tag pw1e [--key s]

Held-out fold probabilities (fit.py), the fixed decodes of score.py, both labellings. Oracles
replace one part of the prediction with the truth and rescore:
  stops   the true stoppages; live frames the prediction called dead take its nearest team
  team    the true team on every frame the prediction calls live (true stoppages not given)
  bounds  the true possessions and stoppages, each possession named by the prediction's majority
          (a tie, or a possession it calls all dead, takes its nearest team at the midpoint)
Then: true possessions matched at IoU 0.5 by length, predicted possessions matched by length, and
what the prediction shows inside each missed true possession.
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
from score import pm, decode  # noqa: E402

H, A, Dd, X = (int(v) for v in pm.label_codes(np.array(list("HADX"))))
BINS = [0, 2, 5, 10, 20, 1e9]
NAMES = ["under 2 s", "2 to 5 s", "5 to 10 s", "10 to 20 s", "over 20 s"]
WHY = ["wrong team", "called dead", "merged into a longer one", "cut up or cut short", "edges off"]


def spell(t, p):
    return pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"]


def fill_team(p):
    """Dead frames take the nearest earlier team label, else the next one."""
    q = p.copy()
    idx = np.where(q != Dd, np.arange(len(q)), -1)
    np.maximum.accumulate(idx, out=idx)
    ok = idx >= 0
    q[ok] = p[idx[ok]]
    if (~ok).any():
        first = np.flatnonzero(p != Dd)
        q[~ok] = p[first[0]] if len(first) else H
    return q


def one(args):
    name, f, key = args
    raw, fold, spans = S.truths(name)
    z = np.load(f)
    out = {}
    for lab, t, dec, pre in (("built", raw, S.RAW, ""), ("folded", fold, S.FOLDED, "f")):
        probs = [z[f"{pre}{key}{i}"] for i in range(len(spans))]
        p = pm.label_codes(decode.labels_from(probs, spans, len(t), **dec))
        g = t != X
        live_t = g & (t != Dd)
        r = {"base": spell(t, p)}
        q = p.copy(); ft = fill_team(p); q[live_t] = ft[live_t]; q[g & (t == Dd)] = Dd
        r["stops"] = spell(t, q)
        q = p.copy(); m = live_t & (p != Dd); q[m] = t[m]
        r["team"] = spell(t, q)
        runs = pm.valid_runs(t)
        gts = pm.possession_spells(t, runs)
        q = t.copy()
        for a, b, k in gts:
            h, aw = (p[a:b] == H).sum(), (p[a:b] == A).sum()
            q[a:b] = ft[(a + b) // 2] if h == aw else (H if h > aw else A)  # a tie takes the prediction's nearest team
        r["bounds"] = spell(t, q)
        prs = pm.possession_spells(p, runs)
        ov = pm.spell_overlaps(gts, prs)
        flags = pm.matched_gt_flags(ov, len(gts), 0.5)
        best = {}
        for iou, i, j in ov:
            best.setdefault(i, (iou, j))
        gl = np.array([(b - a) / 25 for a, b, _ in gts])
        hit = np.zeros((5, 2)); why = np.zeros((5, 5))
        for i, (a, b, k) in enumerate(gts):
            bi = np.searchsorted(BINS, gl[i], "right") - 1
            hit[bi] += (1, flags[i])
            if flags[i]:
                continue
            seg = p[a:b]
            other = A if k == H else H
            if (seg == other).mean() > 0.5:
                w = 0
            elif (seg == Dd).mean() > 0.5:
                w = 1
            elif i in best and (prs[best[i][1]][1] - prs[best[i][1]][0]) > 2 * (b - a):
                w = 2
            elif (seg == k).mean() < 0.7:
                w = 3
            else:
                w = 4
            why[bi, w] += 1
        pmatched = set()
        used = np.zeros(len(gts), bool)
        for iou, i, j in ov:
            if iou < 0.5:
                break
            if used[i] or j in pmatched:
                continue
            used[i] = True; pmatched.add(j)
        ph = np.zeros((5, 2))
        for j, (a, b, k) in enumerate(prs):
            bi = np.searchsorted(BINS, (b - a) / 25, "right") - 1
            ph[bi] += (1, j in pmatched)
        r.update(hit=hit, why=why, phit=ph, n=(len(gts), len(prs)))
        out[lab] = r
    return name, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--key", default="s")
    a = ap.parse_args()
    jobs = []
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(a.tag), fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                jobs.append((os.path.basename(f)[:-4], f, a.key))
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(one, jobs))
    provs = ["idsse", "metrica", "skillcorner"]
    for lab in ("built", "folded"):
        print(f"\n===== labels {lab}, {a.tag} =====")
        print("oracle        " + "".join(f"{p:>12s}" for p in provs) + "        mean")
        for k in ("base", "stops", "team", "bounds"):
            per = [np.mean([r[lab][k] for n, r in res.items() if n.startswith(p)]) for p in provs]
            print(f"{k:14s}" + "".join(f"{x:12.3f}" for x in per) + f"{np.mean(per):12.3f}")
        for p in provs:
            rs = [r[lab] for n, r in res.items() if n.startswith(p)]
            hit = sum(r["hit"] for r in rs); why = sum(r["why"] for r in rs); ph = sum(r["phit"] for r in rs)
            n = np.sum([r["n"] for r in rs], 0)
            print(f"\n{p}: {n[0]} true possessions, {n[1]} predicted")
            print(f"{'true length':14s}{'share':>7s}{'found':>7s}{'missed':>8s}   of the missed: " + ", ".join(WHY))
            for i in range(5):
                if hit[i, 0]:
                    miss = hit[i, 0] - hit[i, 1]
                    print(f"{NAMES[i]:14s}{hit[i, 0] / hit[:, 0].sum():7.2f}{hit[i, 1] / hit[i, 0]:7.2f}{int(miss):8d}   "
                          + "  ".join(f"{w / max(miss, 1):.2f}" for w in why[i]))
            tot = why.sum(0)
            print(f"{'all missed':14s}{'':7s}{hit[:, 1].sum() / hit[:, 0].sum():7.2f}{int(tot.sum()):8d}   " + "  ".join(f"{w / tot.sum():.2f}" for w in tot))
            print("predicted length, share, matched: " + "; ".join(
                f"{NAMES[i]} {ph[i, 0] / ph[:, 0].sum():.2f} {ph[i, 1] / max(ph[i, 0], 1):.2f}" for i in range(5)))
