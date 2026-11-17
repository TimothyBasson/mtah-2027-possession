"""Loss on the failure slice next to the overall score, for saved held-out runs.

    python alldata/contest/slice.py --tags w1e c1e [--key s]

The slice is failure mode 1 of the review (alldata/learned/breakdown.py): a true possession the
prediction gives mostly to the other team. Per tag and label view (as built / folded), mean of
the three providers:

  spellF1   the overall score at score.py's fixed decode
  cost      what the slice costs: spellF1 gained if every such possession were made right
  cost<5    the same for the ones under 5 s
  found<5   share of true possessions under 5 s that follow the other team's possession and are
            matched at IoU 0.5 (the brief steals); wrong<5 the share given to the other team
  spells    predicted possessions per true possession (a net that cuts finer shows up here)
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "learned"))
import score as S  # noqa: E402
from score import pm, decode  # noqa: E402

H, A, Dd, X = (int(v) for v in pm.label_codes(np.array(list("HADX"))))


def one(args):
    name, f, key = args
    raw, fold, spans = S.truths(name)
    z = np.load(f)
    out = []
    for t, dec, pre in ((raw, S.RAW, ""), (fold, S.FOLDED, "f")):
        p = pm.label_codes(decode.labels_from([z[f"{pre}{key}{i}"] for i in range(len(spans))], spans, len(t), **dec))
        fps = 25
        runs = pm.valid_runs(t)
        gts, prs = pm.possession_spells(t, runs), pm.possession_spells(p, runs)
        ov = pm.spell_overlaps(gts, prs)
        found = pm.matched_gt_flags(ov, len(gts), 0.5)
        base = pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"]
        q, q5 = p.copy(), p.copy()
        n5 = f5 = w5 = 0
        for i, (a, b, k) in enumerate(gts):
            other = A if k == H else H
            wrong = (not found[i]) and (p[a:b] == other).mean() > 0.5
            short = (b - a) / fps < 5
            if wrong:
                q[a:b] = k
                if short:
                    q5[a:b] = k
            if short and a > 0 and t[a - 1] == other:
                n5 += 1; f5 += bool(found[i]); w5 += wrong
        g = pm.evaluate(t, q, bootstrap_n=0)["spell_f1_mean"] - base
        g5 = pm.evaluate(t, q5, bootstrap_n=0)["spell_f1_mean"] - base
        out.append([base, g, g5, f5 / max(n5, 1), w5 / max(n5, 1), len(prs) / max(len(gts), 1)])
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--key", default="s")
    a = ap.parse_args()
    provs = ["idsse", "metrica", "skillcorner"]
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for tag in a.tags:
            jobs = [(os.path.basename(f)[:-4], f, a.key) for fold in provs
                    for f in sorted(glob.glob(os.path.join(S.tag_dir(tag), fold, "*.npz"))) if str(np.load(f)["kind"]) == "test"]
            res[tag] = dict(ex.map(one, jobs))
    cols = ["spellF1", "cost", "cost<5", "found<5", "wrong<5", "spells"]
    common = set.intersection(*[set(r) for r in res.values()])
    for vi, view in enumerate(("labels as built", "folded labels")):
        print(f"\n{view} [{len(common)} matches]   " + "  ".join(f"{c:>8s}" for c in cols) + "    spellF1 by provider   cost by provider")
        for tag in a.tags:
            R = {pv: np.array([r[vi] for n, r in res[tag].items() if n.startswith(pv) and n in common]) for pv in provs}
            m = np.mean([R[pv].mean(0) for pv in provs], 0)
            line = f"  {tag:22s}" + "  ".join(f"{x:8.3f}" for x in m)
            line += "    " + " ".join(f"{R[pv][:, 0].mean():.3f}" for pv in provs) + "     " + " ".join(f"{R[pv][:, 1].mean():.3f}" for pv in provs)
            if tag != a.tags[0]:
                b = np.array([res[a.tags[0]][n][vi] for n in sorted(common)])
                c = np.array([res[tag][n][vi] for n in sorted(common)])
                for j, nm in ((0, "F1"), (1, "cost")):
                    dlt = c[:, j] - b[:, j]
                    line += f"   {nm} {dlt.mean():+.3f} (se {dlt.std(ddof=1) / np.sqrt(len(dlt)):.3f}, up {(dlt > 0).sum()}/{len(dlt)})"
            print(line)


if __name__ == "__main__":
    main()
