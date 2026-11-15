"""Held-out spellF1 of saved runs at fixed decodes, on labels as built and folded.

    python alldata/learned/score.py --tags a1 a2 [--kind test] [--key p]

--key p scores the ball as recorded, s the simulated BAR-ATH-like visibility. Same decodes as
deep/fixed.py (cost 4, 1 s / 1 s on labels as built; 2 s / 3 s on folded labels), so numbers are
comparable with the players-only tables. Also accepts deep/ tags by path prefix "deep:bt".
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import data as D  # noqa: E402
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402

RAW = dict(cost=4, min_team=1, min_dead=1)
FOLDED = dict(cost=4, min_team=2, min_dead=3)
LAB = os.path.join(ROOT, "data", "cache", "alldata", "labels")


def truths(name):
    f = os.path.join(LAB, name + ".npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["raw"], z["fold"], [tuple(int(v) for v in h) for h in z["halves"]]
    from control import control_labels
    d = dict(np.load(D.match_path(name)))
    raw, fold = pm.label_codes(d["labels"]), pm.label_codes(control_labels(d))
    os.makedirs(LAB, exist_ok=True)
    np.savez_compressed(f, raw=raw, fold=fold, halves=d["halves"])
    return raw, fold, [tuple(int(v) for v in h) for h in d["halves"]]


def tag_dir(tag):
    if tag.startswith("deep:"):
        return os.path.join(ROOT, "data", "cache", "deep", tag[5:])
    return os.path.join(ROOT, "data", "cache", "alldata", "runs", tag)


def one(args):
    name, files, key, decs = args
    raw, fold, spans = truths(name)
    out = []
    for f in files:
        z = np.load(f)
        k = key if f"{key}0" in z.files else "p"
        probs = [z[f"{k}{i}"] for i in range(len(spans))]
        row = []
        for t, dec in ((raw, decs[0]), (fold, decs[1])):
            r = pm.evaluate(t, pm.label_codes(decode.labels_from(probs, spans, len(t), **dec)), bootstrap_n=0)
            row += [r["spell_f1_mean"], r["alive_dead_f1"], r["team_acc_both_alive"]]
        out.append(row)
    return name, out


def table(tags, kind="test", key="p", decs=(RAW, FOLDED), quiet=False):
    jobs = {}
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(tag_dir(tags[0]), fold, "*.npz"))):
            if str(np.load(f)["kind"]) != kind:
                continue
            name = os.path.basename(f)
            fs = [os.path.join(tag_dir(t), fold, name) for t in tags]
            if all(os.path.exists(x) for x in fs):
                jobs[(fold, name[:-4])] = fs
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, [(n, fs, key, decs) for (_, n), fs in jobs.items()]))
    keys = list(jobs)
    R = np.array([r for _, r in res])
    provs = sorted({n.split("_")[0] for _, n in keys})
    summary = {}
    for k, title in ((0, "spellF1, labels as built"), (3, "spellF1, folded labels"), (2, "team accuracy"), (1, "alive/dead F1")):
        if not quiet:
            print(f"{title}  [{kind}, ball {'as recorded' if key == 'p' else 'simulated visibility'}]")
        for ti, t in enumerate(tags):
            per = [R[[i for i, (_, n) in enumerate(keys) if n.startswith(p)], ti, k].mean() for p in provs]
            summary[(t, k)] = per + [float(np.mean(per))]
            if quiet:
                continue
            line = f"  {t:14s}" + "".join(f"{p[:8]:>9s} {x:.3f}" for p, x in zip(provs, per)) + f"   mean {np.mean(per):.3f}"
            if ti:
                dlt = R[:, ti, k] - R[:, 0, k]
                line += f"   vs {tags[0]}: {dlt.mean():+.3f} (se {dlt.std(ddof=1) / np.sqrt(len(dlt)):.3f}), up {(dlt > 0).sum()}/{len(dlt)}"
            print(line)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--kind", default="test")
    ap.add_argument("--key", default="p")
    a = ap.parse_args()
    table(a.tags, a.kind, a.key)
