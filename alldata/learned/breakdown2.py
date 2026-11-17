"""The other side of breakdown.py: false possessions, timing, stoppages, knock-on misses.

    python alldata/learned/breakdown2.py --tag pw1e [--key s] [--labels folded]

Same saved held-out run and fixed decode as score.py. Prints, per provider:
  1 precision, recall and F1 at each IoU threshold, and how many possessions there are and are drawn
  2 oracle gains: the prediction made right inside every missed possession of one kind
  3 misses by kind and by whether the net itself names the right team inside (mean p >= 0.5)
  4 unmatched predicted possessions by length and by what the truth shows under them
  5 start and end offsets of matched possessions, and the gain from true edges on them
  6 stoppages: frames confused, true stoppages by length and whether they are found
  7 short possessions alone or in a run of short ones
  8 per-match scores
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
LB = [2, 5, 10, 20]
LN = ["under 2 s", "2 to 5 s", "5 to 10 s", "10 to 20 s", "over 20 s"]
HOW = ["wrong team", "called dead", "merged into a longer one", "cut up or cut short", "edges off"]
PK = ["truth is the other team", "truth is a stoppage", "covers several true possessions", "piece of a longer true possession", "edges off"]


def sp(t, p):
    return pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"]


def one(args):
    name, f, key, pre = args
    raw, fold, spans = S.truths(name)
    t = fold if pre == "f" else raw
    dec = S.FOLDED if pre == "f" else S.RAW
    z = np.load(f)
    probs = [z[f"{pre}{key}{i}"] for i in range(len(spans))]
    p = pm.label_codes(decode.labels_from(probs, spans, len(t), **dec))
    P = np.full((len(t), 3), 1 / 3)
    for i, (a, b) in enumerate(spans):
        P[a:b + 1] = decode.upsample(probs[i], b - a + 1)
    runs = pm.valid_runs(t)
    gts, prs = pm.possession_spells(t, runs), pm.possession_spells(p, runs)
    ov = pm.spell_overlaps(gts, prs)
    R = {"base": sp(t, p), "n": (len(gts), len(prs))}
    R["iou"] = [pm.match_spells(ov, len(gts), len(prs), th) for th in pm.SPELL_IOUS]
    flags = pm.matched_gt_flags(ov, len(gts), 0.5)
    pair, usedp = {}, set()
    for iou, i, j in ov:
        if iou < 0.5:
            break
        if i in pair or j in usedp:
            continue
        pair[i] = (j, iou); usedp.add(j)
    best = {}
    for iou, i, j in ov:
        best.setdefault(i, (iou, j))
    bestp = {}
    for iou, i, j in ov:
        bestp.setdefault(j, (iou, i))
    # 2, 3: misses by kind
    kind = np.full(len(gts), -1)
    conf = np.zeros(len(gts), bool)
    glen = np.array([(b - a) / 25 for a, b, _ in gts])
    for i, (a, b, k) in enumerate(gts):
        conf[i] = P[a:b, 0 if k == H else 1].mean() >= 0.5
        if flags[i]:
            continue
        seg = p[a:b]
        other = A if k == H else H
        if (seg == other).mean() > 0.5:
            kind[i] = 0
        elif (seg == Dd).mean() > 0.5:
            kind[i] = 1
        elif i in best and (prs[best[i][1]][1] - prs[best[i][1]][0]) > 2 * (b - a):
            kind[i] = 2
        elif (seg == k).mean() < 0.7:
            kind[i] = 3
        else:
            kind[i] = 4
    lb = np.searchsorted(LB, glen, "right")
    R["miss"] = np.array([[[((kind == w) & (lb == l) & (conf == c)).sum() for c in (0, 1)] for l in range(5)] for w in range(5)])
    R["glen"] = np.array([[(lb == l).sum(), ((lb == l) & flags).sum()] for l in range(5)])
    og = {}
    for w in range(5):
        for sel, nm in ((kind == w, "all"), ((kind == w) & (glen < 5), "short")):
            q = p.copy()
            for i in np.flatnonzero(sel):
                a, b, k = gts[i]; q[a:b] = k
            og[(w, nm)] = sp(t, q) - R["base"]
    for c, nm in ((0, "net wrong"), (1, "net right")):
        q = p.copy()
        for i in np.flatnonzero((kind >= 0) & (conf == c)):
            a, b, k = gts[i]; q[a:b] = k
        og[nm] = sp(t, q) - R["base"]
    q = p.copy()
    for i in np.flatnonzero(kind >= 0):
        a, b, k = gts[i]; q[a:b] = k
    og["every miss"] = sp(t, q) - R["base"]
    R["og"] = og
    # 4: unmatched predictions
    pk = np.zeros((5, 5))
    plen = np.zeros((5, 2))
    for j, (a, b, k) in enumerate(prs):
        l = int(np.searchsorted(LB, (b - a) / 25, "right"))
        plen[l] += (1, j in usedp)
        if j in usedp:
            continue
        seg = t[a:b]
        other = A if k == H else H
        if (seg == other).mean() > 0.5:
            w = 0
        elif (seg == Dd).mean() > 0.5 or (seg == X).mean() > 0.5:
            w = 1
        elif j in bestp and (b - a) > 2 * (gts[bestp[j][1]][1] - gts[bestp[j][1]][0]):
            w = 2
        elif j in bestp and (b - a) < 0.5 * (gts[bestp[j][1]][1] - gts[bestp[j][1]][0]):
            w = 3
        else:
            w = 4
        pk[w, l] += 1
    R["pk"], R["plen"] = pk, plen
    # 5: timing of matched pairs
    offs = []
    q = p.copy()
    for i, (j, iou) in pair.items():
        a, b, k = gts[i]; pa, pb, _ = prs[j]
        offs.append(((pa - a) / 25, (pb - b) / 25, iou, glen[i]))
    R["offs"] = np.array(offs).reshape(-1, 4)
    for i, (j, iou) in pair.items():  # true edges on matched possessions: paint truth over the union of both
        a, b, k = gts[i]; pa, pb, _ = prs[j]
        lo, hi = min(a, pa), max(b, pb)
        q[lo:hi] = np.where(t[lo:hi] == X, q[lo:hi], t[lo:hi])
    R["edge_gain"] = sp(t, q) - R["base"]
    # 6: stoppages
    g = t != X
    R["fr"] = np.array([[((t == u) & (p == v)).sum() for v in (H, A, Dd)] for u in (H, A, Dd)])
    segs = [(a, b) for a, b, k in pm.segments(t, runs) if k == Dd]
    sb = np.zeros((4, 2))
    for a, b in segs:
        l = int(np.searchsorted([5, 15, 40], (b - a) / 25, "right"))
        sb[l] += (1, (p[a:b] == Dd).mean() > 0.5)
    R["stop"] = sb
    psegs = [(a, b) for a, b, k in pm.segments(p, runs) if k == Dd]
    R["pstop"] = np.array([len(psegs), sum((t[a:b] == Dd).mean() < 0.5 for a, b in psegs)])
    q = p.copy(); q[g & (t == Dd)] = Dd
    m = g & (t != Dd) & (p == Dd)
    idx = np.where(p != Dd, np.arange(len(p)), -1); np.maximum.accumulate(idx, out=idx)
    q[m & (idx >= 0)] = p[idx[m & (idx >= 0)]]
    R["stop_gain"] = sp(t, q) - R["base"]
    # 7: short possessions alone or in a run of short ones
    ch = np.zeros((3, 2))
    for i in np.flatnonzero(glen < 5):
        nb = sum(1 for d in (-1, 1) if 0 <= i + d < len(gts) and glen[i + d] < 5 and
                 (gts[i + d][0] == gts[i][1] or gts[i + d][1] == gts[i][0]))
        ch[nb] += (1, flags[i])
    R["chain"] = ch
    q = p.copy()
    for i in np.flatnonzero(glen < 5):
        nb = sum(1 for d in (-1, 1) if 0 <= i + d < len(gts) and glen[i + d] < 5 and
                 (gts[i + d][0] == gts[i][1] or gts[i + d][1] == gts[i][0]))
        if nb:
            a, b, k = gts[i]; q[a:b] = k
    R["chain_gain"] = sp(t, q) - R["base"]
    return name, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--key", default="s")
    ap.add_argument("--labels", default="folded", choices=["folded", "built"])
    a = ap.parse_args()
    pre = "f" if a.labels == "folded" else ""
    jobs = []
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(a.tag), fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                jobs.append((os.path.basename(f)[:-4], f, a.key, pre))
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(one, jobs))
    provs = ["idsse", "metrica", "skillcorner"]
    by = {pv: [r for n, r in res.items() if n.startswith(pv)] for pv in provs}
    mean = lambda fn: [float(np.mean([fn(r) for r in by[pv]])) for pv in provs]  # noqa: E731
    tot = lambda key: [np.sum([r[key] for r in by[pv]], 0) for pv in provs]  # noqa: E731
    fmt = lambda xs, f="{:.3f}": " / ".join(f.format(x) for x in xs) + f"  mean {np.mean(xs):.3f}"  # noqa: E731
    print(f"== {a.tag}, labels {a.labels}; IDSSE / Metrica / SkillCorner")
    print("1 base", fmt(mean(lambda r: r["base"])))
    n = tot("n"); print("  true possessions", [int(x[0]) for x in n], "drawn", [int(x[1]) for x in n])
    io = tot("iou")
    for ti, th in enumerate(pm.SPELL_IOUS):
        print(f"  IoU {th}: " + "  ".join(f"P {x[ti][0] / (x[ti][0] + x[ti][1]):.2f} R {x[ti][0] / (x[ti][0] + x[ti][2]):.2f}" for x in io))
    print("2 oracle gain, all lengths (and under 5 s only)")
    for w, nm in enumerate(HOW):
        print(f"  {nm:26s}", fmt(mean(lambda r: r["og"][(w, "all")]), "{:+.3f}"), "  (", fmt(mean(lambda r: r["og"][(w, "short")]), "{:+.3f}"), ")")
    for nm in ("net wrong", "net right", "every miss"):
        print(f"  {nm:26s}", fmt(mean(lambda r: r["og"][nm]), "{:+.3f}"))
    print("3 misses: kind x length, counts 'net wrong inside+net right inside'")
    ms = tot("miss"); gl = tot("glen")
    for w, nm in enumerate(HOW):
        print(f"  {nm:26s}" + " | ".join(" ".join(f"{int(m[w, l, 0]):4d}+{int(m[w, l, 1]):<4d}" for l in range(5)) for m in ms))
    print("  possessions / found       " + " | ".join(" ".join(f"{int(g[l, 0]):4d}/{int(g[l, 1]):<4d}" for l in range(5)) for g in gl))
    print("4 predicted possessions by length: drawn / matched;  then unmatched by kind x length")
    pl = tot("plen"); pk = tot("pk")
    print("  drawn/matched             " + " | ".join(" ".join(f"{int(g[l, 0]):4d}/{int(g[l, 1]):<4d}" for l in range(5)) for g in pl))
    for w, nm in enumerate(PK):
        print(f"  {nm:34s}" + " | ".join(" ".join(f"{int(m[w, l]):4d}" for l in range(5)) for m in pk))
    print("5 matched possessions: start offset and end offset in s (pred - true): 10/25/50/75/90 pct; |offset| > 1 s share; mean IoU")
    for pv in provs:
        o = np.vstack([r["offs"] for r in by[pv]])
        print(f"  {pv:12s} start", np.round(np.percentile(o[:, 0], [10, 25, 50, 75, 90]), 2), f"{(abs(o[:, 0]) > 1).mean():.2f}",
              " end", np.round(np.percentile(o[:, 1], [10, 25, 50, 75, 90]), 2), f"{(abs(o[:, 1]) > 1).mean():.2f}", f" IoU {o[:, 2].mean():.2f}")
    print("  true edges on matched possessions:", fmt(mean(lambda r: r["edge_gain"]), "{:+.3f}"))
    print("6 frames, rows truth H A D, columns prediction H A D (share of row)")
    for pv, m in zip(provs, tot("fr")):
        print(f"  {pv:12s}", np.round(m / m.sum(1, keepdims=True), 3).tolist(), " dead share true", round(m[2].sum() / m.sum(), 3), "pred", round(m[:, 2].sum() / m.sum(), 3))
    print("  true stoppages under 5 s, 5-15, 15-40, over 40: n / called dead")
    for pv, m in zip(provs, tot("stop")):
        print(f"  {pv:12s}", " ".join(f"{int(x[0])}/{int(x[1])}" for x in m))
    print("  predicted stoppages / of them mostly live in truth:", [tuple(int(v) for v in x) for x in tot("pstop")])
    print("  true stoppages given:", fmt(mean(lambda r: r["stop_gain"]), "{:+.3f}"))
    print("7 under-5 s possessions with 0, 1, 2 touching under-5 s neighbours: n / found")
    for pv, m in zip(provs, tot("chain")):
        print(f"  {pv:12s}", " ".join(f"{int(x[0])}/{int(x[1])} ({x[1] / max(x[0], 1):.2f})" for x in m))
    print("  made right where they touch another short one:", fmt(mean(lambda r: r["chain_gain"]), "{:+.3f}"))
    print("8 per match")
    for pv in provs:
        print(f"  {pv:12s}", " ".join(f"{r['base']:.2f}" for n_, r in sorted(res.items()) if n_.startswith(pv)))


if __name__ == "__main__":
    main()
