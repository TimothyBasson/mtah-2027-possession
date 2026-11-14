"""Held-out spellF1 of saved fold probabilities at one fixed decode, on the labels as built and
on the labels folded to the controlled-possession rule (external/control.py, 2 s / 3 s minimum).

    python deep/fixed.py --tags v2 st [--kind test]

No decode is picked on validation matches, so tags differ only by their networks. Prints each
provider's mean, the equal-provider mean, and per-match differences from the first tag.
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402
from control import control_labels  # noqa: E402

RAW = dict(cost=4, min_team=1, min_dead=1)
FOLDED = dict(cost=4, min_team=2, min_dead=3)


def one(args):
    name, files = args
    d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
    spans = [tuple(int(v) for v in h) for h in d["halves"]]
    truths = (pm.label_codes(d["labels"]), pm.label_codes(control_labels(d)))
    out = []
    for f in files:
        z = np.load(f)
        probs = [z[f"p{i}"] for i in range(len(spans))]
        row = []
        for t, dec in zip(truths, (RAW, FOLDED)):
            r = pm.evaluate(t, pm.label_codes(decode.labels_from(probs, spans, len(t), **dec)), bootstrap_n=0)
            row += [r["spell_f1_mean"], r["alive_dead_f1"], r["team_acc_both_alive"]]
        out.append(row)
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--kind", default="test")
    a = ap.parse_args()
    base = os.path.join(ROOT, "data", "cache", "deep")
    jobs = {}
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(base, a.tags[0], fold, "*.npz"))):
            if str(np.load(f)["kind"]) != a.kind:
                continue
            name = os.path.basename(f)
            fs = [os.path.join(base, t, fold, name) for t in a.tags]
            if all(os.path.exists(x) for x in fs):
                jobs[(fold, name[:-4])] = fs
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(one, [(n, fs) for (_, n), fs in jobs.items()]))
    keys = list(jobs)
    R = np.array([r for _, r in res])  # match, tag, 6
    provs = sorted({n.split("_")[0] for _, n in keys})
    for k, title in ((0, "spellF1, labels as built"), (3, "spellF1, folded labels"), (5, "team accuracy"), (4, "alive/dead F1")):
        print(title)
        for ti, t in enumerate(a.tags):
            per = [R[[i for i, (_, n) in enumerate(keys) if n.startswith(p)], ti, k].mean() for p in provs]
            line = f"  {t:10s}" + "".join(f"{p[:8]:>9s} {x:.3f}" for p, x in zip(provs, per)) + f"   mean {np.mean(per):.3f}"
            if ti:
                dlt = R[:, ti, k] - R[:, 0, k]
                line += f"   per match vs {a.tags[0]}: {dlt.mean():+.3f} (se {dlt.std(ddof=1) / np.sqrt(len(dlt)):.3f}), up on {(dlt > 0).sum()} of {len(dlt)}"
            print(line)


if __name__ == "__main__":
    main()
