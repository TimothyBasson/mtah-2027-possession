"""Checks asked for in review: is the model better than the players-only prior, and where.

    python alldata/physics/review.py paired     # s5h against the prior, paired by match, bootstrap
    python alldata/physics/review.py split      # ball shown against ball hidden, simulated matches
    python alldata/physics/review.py bar_ath    # the BAR-ATH file against the players-only file

"paired" resamples matches inside each provider (the mean of the three providers is the figure),
2000 times. "split" gives frame accuracy and the share of true possessions matched at IoU 0.5,
by how much of the possession had the ball in view. "bar_ath" needs no labels except for its
last block, which is the public 5 minutes.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import decode as DC  # noqa: E402
from build import table_path  # noqa: E402
from train import pred_dir  # noqa: E402
import possession_metrics as pm  # noqa: E402

DEC = dict(cost=4.0, min_team=1.0, min_dead=1.0)
RUNS = ("prior", "s5h")
FPS = 25


def one(name):
    z = np.load(table_path(name, "barath", "f5"))
    lab, fol, shown = z["labels"], z["folded"], z["shown"].astype(bool)
    spans = [tuple(int(v) for v in h) for h in z["halves"]]
    t, tf = pm.label_codes(lab), pm.label_codes(fol)
    runs = pm.valid_runs(t)
    gs = np.asarray(pm.possession_spells(t, runs))
    seen = np.array([shown[a:b].mean() for a, b in gs[:, :2]])
    out = {}
    for run in RUNS:
        q = np.load(os.path.join(pred_dir(run), name + ".npz"))
        pred = DC.labels_from([q[f"p{i}"] for i in range(len(spans))], spans, len(lab), **DEC)
        p = pm.label_codes(pred)
        sc = [pm.evaluate(x, p, bootstrap_n=0)["spell_f1_mean"] for x in (t, tf)]
        graded = lab != "X"
        acc = [(pred[graded & m] == lab[graded & m]).mean() for m in (shown, ~shown)]
        ps = pm.possession_spells(pm.fill_abstentions(p, runs), runs)
        flags = np.asarray(pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5))
        by = []
        for m in (seen >= 0.8, (seen > 0.2) & (seen < 0.8), seen <= 0.2):
            by += [m.sum(), flags[m].sum()]
        out[run] = (sc, acc, by, [(graded & shown).sum(), (graded & ~shown).sum()])
    return name, out


def gather():
    with ProcessPoolExecutor(2) as ex:
        return dict(ex.map(one, C.names()))


def paired(res):
    names = list(res)
    prov = np.array([C.PROVIDERS.index(C.provider(n)) for n in names])
    rng = np.random.default_rng(0)
    for k, title in ((0, "as given"), (1, "folded")):
        a = np.array([res[n]["prior"][0][k] for n in names])
        b = np.array([res[n]["s5h"][0][k] for n in names])
        mean = lambda v, idx: np.mean([v[idx[prov[idx] == p]].mean() for p in range(3)])  # noqa: E731
        every = np.arange(len(names))
        draws = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(every[prov == p], (prov == p).sum()) for p in range(3)])
            draws.append((mean(a, idx), mean(b, idx)))
        d = np.array(draws)
        lo = lambda v: np.percentile(v, 2.5)  # noqa: E731
        hi = lambda v: np.percentile(v, 97.5)  # noqa: E731
        diff = d[:, 1] - d[:, 0]
        print(f"{title}: prior {mean(a, every):.3f} [{lo(d[:, 0]):.3f}, {hi(d[:, 0]):.3f}]   "
              f"s5h {mean(b, every):.3f} [{lo(d[:, 1]):.3f}, {hi(d[:, 1]):.3f}]   "
              f"s5h - prior {mean(b, every) - mean(a, every):+.3f} [{lo(diff):+.3f}, {hi(diff):+.3f}], "
              f"s5h ahead on {(b > a).sum()} of {len(names)} matches, in {(diff > 0).mean():.0%} of resamples")
        for p, pn in enumerate(C.PROVIDERS):
            m = prov == p
            print(f"    {pn:12s} {(b - a)[m].mean():+.3f}, ahead on {(b > a)[m].sum()} of {m.sum()}")


def split(res):
    print("frame accuracy (H/A/D), then true possessions matched at IoU 0.5 by share of the possession with the ball in view")
    for p in C.PROVIDERS:
        r = [res[n] for n in res if C.provider(n) == p]
        n_sh = np.sum([x["s5h"][3] for x in r], 0)
        print(f"{p}: ball shown on {n_sh[0] / n_sh.sum():.0%} of graded frames")
        for run in RUNS:
            w = np.array([x[run][3] for x in r], float)
            acc = np.nansum(np.array([x[run][1] for x in r]) * w, 0) / w.sum(0)
            by = np.sum([x[run][2] for x in r], 0)
            print(f"    {run:6s} accuracy shown {acc[0]:.3f}, hidden {acc[1]:.3f};   matched: mostly shown "
                  f"{by[1] / max(by[0], 1):.2f} of {by[0]}, mixed {by[3] / max(by[2], 1):.2f} of {by[2]}, "
                  f"mostly hidden {by[5] / max(by[4], 1):.2f} of {by[4]}")


def runs_of(lab):
    edges = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
    return [(a, b, lab[a]) for a, b in zip(edges[:-1], edges[1:])]


def stats(lab, a, b):
    seg = lab[a:b + 1]
    rs = runs_of(seg)
    poss = [r for r in rs if r[2] in "HA"]
    dead = [r for r in rs if r[2] == "D"]
    ln = np.array([r[1] - r[0] for r in poss]) / FPS
    return (f"{len(poss)} possessions, mean {ln.mean():.1f} s, median {np.median(ln):.1f} s, {np.mean(ln < 2):.0%} under 2 s; "
            f"{len(dead)} stoppages, D {np.mean(seg == 'D'):.0%}; runs under 1 s: {sum(r[1] - r[0] < FPS for r in rs)}; "
            f"changes per minute {len(rs) / (len(seg) / FPS / 60):.1f}")


def bar_ath():
    import pandas as pd
    res_dir = os.path.join(C.ROOT, "evaluation", "results")
    phys = pd.read_csv(os.path.join(res_dir, "alldata_physics.csv")).label.values.astype("<U1")
    bt = pd.read_csv(os.path.join(res_dir, "deep_bt.csv")).label.values.astype("<U1")
    z = np.load(table_path("bar_ath", "barath", "f5"))
    shown = z["shown"].astype(bool)
    halves = [tuple(int(v) for v in h) for h in z["halves"]]
    sys.path.insert(0, C.ROOT)
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    kicks = list(KICKOFF_VIDEO_FRAMES.values())
    graded = np.zeros(len(phys), bool)
    for (a, b), k in zip(halves, kicks):
        graded[k:b + 1] = True
        print(f"half from frame {k} to {b}: ball in view {shown[k:b + 1].mean():.0%}")
        print("    this file    ", stats(phys, k, b))
        print("    players-only ", stats(bt, k, b))
    # where the two files differ
    diff = graded & (phys != bt)
    print(f"the files differ on {diff.sum()} of {graded.sum()} frames after kick-off ({diff.sum() / graded.sum():.1%})")
    kinds = {"team against team": np.isin(phys, ("H", "A")) & np.isin(bt, ("H", "A")),
             "this file dead, players-only alive": (phys == "D") & (bt != "D"),
             "this file alive, players-only dead": (phys != "D") & (bt == "D")}
    for k, m in kinds.items():
        m = m & diff
        print(f"    {k:36s} {m.sum():6d} frames ({m.sum() / diff.sum():.0%}), ball in view on {shown[m].mean():.0%} of them")
    print(f"    differing share where the ball is in view {diff[graded & shown].mean():.1%}, where it is not "
          f"{diff[graded & ~shown].mean():.1%}")
    segs = sorted(((b - a, a, b) for a, b, v in runs_of(diff.astype(np.int8).astype("<U1")) if v == "1"), reverse=True)[:20]
    print("longest 20 stretches of disagreement: start frame, seconds, this file / players-only, ball in view")
    for n, a, b in segs:
        top = lambda x: "".join(sorted(set(x[a:b]), key=lambda c: -(x[a:b] == c).sum()))  # noqa: E731
        print(f"    {a:7d} {n / FPS:5.1f} s  {top(phys):3s} / {top(bt):3s}  {shown[a:b].mean():.0%}")
    # tagged set pieces explained by a stoppage
    sys.path.insert(0, os.path.join(C.ROOT, "stops"))
    import tags as TG
    starts = [s for s in TG.set_pieces() if graded[s]]
    for nm, lab in (("this file", phys), ("players-only", bt)):
        hit = sum((lab[max(0, s + int(TG.NEAR[0] * FPS)):s + int(TG.NEAR[1] * FPS)] == "D").any() for s in starts)
        n_dead = sum(1 for a, b, v in runs_of(lab) if v == "D" and graded[a])
        print(f"{nm}: {n_dead} stoppages against {len(starts)} tagged set pieces, {hit} of them beside a stoppage")
    # the public 5 minutes (a check, nothing is chosen on it)
    gt = pd.read_csv(os.path.join(C.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")).label.values.astype("<U1")
    w = gt != "X"
    print(f"public 5 minutes: ball in view {shown[w].mean():.0%}; D share truth {np.mean(gt[w] == 'D'):.0%}, "
          f"this file {np.mean(phys[w] == 'D'):.0%}, players-only {np.mean(bt[w] == 'D'):.0%}; "
          f"dead share of ball-in-view frames (truth) {np.mean(gt[w & shown] == 'D'):.0%}")
    for nm, lab in (("this file", phys), ("players-only", bt)):
        print(f"    {nm:13s} frame accuracy ball in view {(lab[w & shown] == gt[w & shown]).mean():.3f} "
              f"({(w & shown).sum()} frames), not in view {(lab[w & ~shown] == gt[w & ~shown]).mean():.3f} ({(w & ~shown).sum()})")
    for nm, lab in (("this file", phys), ("players-only", bt)):
        g = graded
        print(f"    {nm:13s} whole match: dead share of ball-in-view frames {np.mean(lab[g & shown] == 'D'):.0%}, "
              f"of ball-missing frames {np.mean(lab[g & ~shown] == 'D'):.0%}")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "bar_ath":
        bar_ath()
    else:
        res = gather()
        paired(res) if step == "paired" else split(res)
        if step == "paired":
            split(res)
