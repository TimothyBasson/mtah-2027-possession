"""Two things the review of #29 asked for. Neither reads a label of BAR-ATH.

    python deep/compare.py --against evaluation/results/hmm_v7.csv   # deep v2 against another file
    python deep/compare.py --sweep                                   # decode settings

--against: on BAR-ATH's graded frames, possessions per half for each file, how often the two
agree with the ball seen and with it missing, and what kind of disagreement it is.
--sweep: every decode setting, the same in all three folds (nothing picked), with its held-out
spellF1 on the outside matches beside the number of possessions it gives on BAR-ATH.
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation"), ROOT]
import decode  # noqa: E402


def runs(x):
    return [(k, len(list(g))) for k, g in itertools.groupby(x)]


def against(mine, other):
    import load
    import pandas as pd
    players, ball, team, halves = load.load()
    a, b = (pd.read_csv(p).label.to_numpy() for p in (mine, other))
    seen = np.isfinite(ball[:, 0])
    g = np.zeros(len(a), bool)
    for i, (s, e) in enumerate(halves):
        g[s:e + 1] = True
        for name, x in ((mine, a), (other, b)):
            r = [n for k, n in runs(x[s:e + 1]) if k in "HA"]
            print(f"half {i + 1}  {os.path.basename(name):14s} {len(r):4d} possessions, median {np.median(r) / 25:4.1f} s, "
                  f"mean {np.mean(r) / 25:4.1f} s, dead {np.mean(x[s:e + 1] == 'D'):.0%}")
    for name, m in (("all graded frames", g), ("ball seen", g & seen), ("ball missing", g & ~seen)):
        print(f"{name:18s} {m.sum():7d} frames ({m.sum() / g.sum():.0%})  agree {np.mean(a[m] == b[m]):.3f}")
    live = g & np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"])
    kinds = {"other team (both live)": live & (a != b), "mine dead, other live": g & (a == "D") & (b != "D"),
             "mine live, other dead": g & (a != "D") & (b == "D")}
    for name, m in kinds.items():
        print(f"{name:24s} {m.sum() / g.sum():6.1%} of graded frames; ball missing on {np.mean(~seen[m]):.0%} of them")
    print(f"team agreement where both live: {np.mean(a[live] == b[live]):.3f}"
          f" (ball seen {np.mean(a[live & seen] == b[live & seen]):.3f}, missing {np.mean(a[live & ~seen] == b[live & ~seen]):.3f})")


def sweep(tag):
    import possession_metrics as pm
    grid = [dict(cost=c, min_team=t, min_dead=d) for t, d in ((0, 0), (0.5, 0.5), (1, 1), (2, 3)) for c in (0, 1, 2, 4, 8)]
    base = os.path.join(ROOT, "data", "cache", "deep", tag)
    import load
    z = np.load(os.path.join(base, f"bar_ath_probs{load.cache_suffix()}.npz"))  # written by deep/run.py
    halves = load.load()[3]
    n = max(e for _, e in halves) + 1
    scores = {}
    for fold in ("idsse", "metrica", "skillcorner"):
        rows = []
        for f in sorted(glob.glob(os.path.join(base, fold, fold + "_*.npz"))):
            p = np.load(f)
            d = np.load(os.path.join(ROOT, "data", "external", os.path.basename(f)))
            spans = [tuple(h) for h in d["halves"]]
            truth = pm.label_codes(d["labels"])
            probs = [p[f"p{i}"] for i in range(len(spans))]
            rows.append([pm.evaluate(truth, pm.label_codes(decode.labels_from(probs, spans, len(truth), **c)),
                                     bootstrap_n=0)["spell_f1_mean"] for c in grid])
        scores[fold] = np.mean(rows, 0)
        print(fold, "done", flush=True)
    print(f"{'cost':>4s} {'min H/A':>7s} {'min D':>5s}   IDSSE  Metrica  SkillC    mean   BAR-ATH possessions (median s)")
    for i, c in enumerate(grid):
        pred = decode.labels_from([z["p0"], z["p1"]], halves, n, **c)
        r = [m for (s, e) in halves for k, m in runs(pred[s:e + 1]) if k in "HA"]
        s = [scores[f][i] for f in ("idsse", "metrica", "skillcorner")]
        print(f"{c['cost']:4g} {c['min_team']:7g} {c['min_dead']:5g}   " + "   ".join(f"{x:.3f}" for x in s)
              + f"   {np.mean(s):.3f}   {len(r):4d} ({np.median(r) / 25:.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--mine", default=os.path.join(ROOT, "evaluation", "results", "deep_v2.csv"))
    ap.add_argument("--against")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.against:
        against(a.mine, a.against)
    if a.sweep:
        sweep(a.tag)
