"""Decode settings and averages of trained copies, held out, on the controlled-possession labels.

    python deep/variants.py [--jobs 4]

A source is one or more sets of held-out probabilities (a tag under data/cache/deep, or a folder
laid out the same way), averaged. Each source is decoded at every switch cost and every pair of
shortest possession / shortest stoppage, with the same setting on every match. Scored against the
outside labels as they are and folded towards the controlled-possession rule (external/control.py).
Then, for each source, the setting each provider would get if it were picked on the other two
providers' folded scores: that row is the held-out number.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external"), ROOT]
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402
from control import control_labels  # noqa: E402

SHARED = "/mnt/project-files/deep/"
SOURCES = {"v2": ["v2"], "v2 + seed 1": ["v2", SHARED + "v2s1"], "v4": ["v4"], "v2 + v4": ["v2", "v4"],
           "v2 + seed 1 + v4": ["v2", SHARED + "v2s1", "v4"],
           "bt": [SHARED + "bt"], "bt + seed 1": [SHARED + "bt", SHARED + "bts1"],
           "bt + seed 1 + v2 + seed 1": [SHARED + "bt", SHARED + "bts1", "v2", SHARED + "v2s1"],
           "all five": [SHARED + "bt", SHARED + "bts1", "v2", SHARED + "v2s1", "v4"]}
GRID = [dict(cost=c, min_team=t, min_dead=d) for c in (2, 4, 8) for t, d in ((1, 1), (2, 2), (2, 3), (3, 3))]
SHIPPED = GRID.index(dict(cost=4, min_team=1, min_dead=1))
PROVIDERS = ("idsse", "metrica", "skillcorner")


def folder(tag):
    return tag if os.path.isabs(tag) else os.path.join(ROOT, "data", "cache", "deep", tag)


def one(path):
    name = os.path.basename(path)[:-4]
    d = np.load(path)
    spans = [tuple(int(v) for v in h) for h in d["halves"]]
    views = [pm.label_codes(v) for v in (d["labels"], control_labels(d))]
    out = {}
    for s, tags in SOURCES.items():
        files = [os.path.join(folder(t), name.split("_")[0], name + ".npz") for t in tags]
        if not all(os.path.exists(f) for f in files):
            continue
        zs = [np.load(f) for f in files]
        probs = [np.mean([z[f"p{i}"] for z in zs], 0) for i in range(len(spans))]
        rows = []
        for c in GRID:
            pred = decode.labels_from(probs, spans, len(d["labels"]), **c)
            pred[d["labels"] == "X"] = "X"
            code = pm.label_codes(pred)
            rows.append([pm.evaluate(v, code, bootstrap_n=0)["spell_f1_mean"] for v in views])
        out[s] = rows
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(PROVIDERS)]
    with Pool(a.jobs) as pool:
        rows = dict(pool.imap_unordered(one, paths))
    names = sorted(rows)
    name_of = lambda c: f"cost {c['cost']}, min {c['min_team']}/{c['min_dead']} s"  # noqa: E731
    held = []
    for s in SOURCES:
        if not all(s in rows[n] for n in names):
            print(f"\n{s}: probabilities missing, skipped")
            continue
        S = {k: np.array([rows[n][s] for n in names if n.startswith(k)]) for k in PROVIDERS}  # (matches, grid, view)
        print(f"\n{s}\n  {'setting':22s} raw: IDSSE Metrica SkillC   mean | folded: IDSSE Metrica SkillC   mean")
        for j, c in enumerate(GRID):
            line = f"  {name_of(c):22s}"
            for v in (0, 1):
                m = [S[k][:, j, v].mean() for k in PROVIDERS]
                line += "      " + "   ".join(f"{x:.3f}" for x in m) + f"   {np.mean(m):.3f}" + (" |" if v == 0 else "")
            print(line + ("   <- shipped" if j == SHIPPED else ""))
        got, picks = [], []
        for k in PROVIDERS:
            j = int(np.argmax(np.mean([S[o][:, :, 1].mean(0) for o in PROVIDERS if o != k], 0)))
            got.append((S[k][:, j, 0].mean(), S[k][:, j, 1].mean()))
            picks.append(name_of(GRID[j]))
        g = np.mean(got, 0)
        held.append((s, [x[1] for x in got], g[1], [x[0] for x in got], g[0], picks))
    print("\nheld out, setting picked on the other two providers' folded scores")
    print(f"  {'source':28s} folded: IDSSE Metrica SkillC   mean | raw: IDSSE Metrica SkillC   mean   picks")
    for s, f, fm, r, rm, picks in held:
        print(f"  {s:28s}         " + "   ".join(f"{x:.3f}" for x in f) + f"   {fm:.3f} |      "
              + "   ".join(f"{x:.3f}" for x in r) + f"   {rm:.3f}   " + "; ".join(picks))


if __name__ == "__main__":
    main()
