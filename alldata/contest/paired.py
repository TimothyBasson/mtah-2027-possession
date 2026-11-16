"""Paired per-match differences with the submission's touch decode, with a bootstrap over matches.

    python alldata/contest/paired.py --base pw1e --tags pc1e pcr1e [--seeds pcr1 pcr1s1]

For each tag: spellF1 (as built / folded) per match with alldata/touch's evidence (threshold 0.3,
weight 1, hold 3 s), minus the base's on the same match. Prints the mean difference per provider
and of the provider means, with a 90% interval from resampling matches within each provider
(2,000 draws), and the share of matches up. --seeds A B prints the same for two seeds of one
recipe against each other: the size of a difference that seed noise alone makes.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "touch"))
import common as C  # noqa: E402
import tuse  # noqa: E402

SET = (0.3, 1.0, 3.0, 0)


def scores(tag):
    ms = tuse.matches(tag, "test")
    res = dict(C.pool_map(tuse.one, [(fo, n, tag, "t1", 1, [SET[1:]], [SET[0]]) for fo, n in ms], 4))
    return {n: np.array([r[SET]["raw"], r[SET]["fold"]], float) for (fo, n), r in res.items()}


def show(title, d):
    rng = np.random.default_rng(0)
    provs = C.PROVIDERS
    per = {pv: np.array([v for n, v in d.items() if n.startswith(pv)]) for pv in provs}
    boot = np.array([np.mean([per[pv][rng.integers(len(per[pv]), size=len(per[pv]))].mean(0) for pv in provs], 0) for _ in range(2000)])
    mean = np.mean([per[pv].mean(0) for pv in provs], 0)
    lo, hi = np.percentile(boot, [5, 95], axis=0)
    up = np.mean([v > 0 for v in d.values()], 0)
    line = f"  {title:24s}"
    for j, nm in enumerate(("built", "folded")):
        line += f" | {nm} {mean[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}] up {up[j]:.0%} (" + " ".join(f"{per[pv][:, j].mean():+.3f}" for pv in provs) + ")"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pw1e")
    ap.add_argument("--tags", nargs="+", default=[])
    ap.add_argument("--seeds", nargs=2, action="append", default=[])
    a = ap.parse_args()
    print("difference in spellF1 with touch evidence, mean of provider means [90% bootstrap over matches], share of matches up (IDSSE Metrica SkillCorner)")
    if a.tags:
        b = scores(a.base)
        for t in a.tags:
            s = scores(t)
            show(f"{t} - {a.base}", {n: s[n] - b[n] for n in b if n in s})
    for x, y in a.seeds:
        sx, sy = scores(x), scores(y)
        show(f"{y} - {x} (seeds)", {n: sy[n] - sx[n] for n in sx if n in sy})


if __name__ == "__main__":
    main()
