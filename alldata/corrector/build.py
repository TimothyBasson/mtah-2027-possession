"""Corrector rows for every outside match: data/cache/alldata/corrector/feats/<match>.npz.

    python alldata/corrector/build.py [--base pw1e] [--version 1] [--key s]

Each match takes the main model's probabilities from the fold in which its provider was held out
(run `--base`), and the ball of the same simulated version those probabilities were made on.
"""

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import cfeats

import data as D

OUT = os.path.join(ROOT, "data", "cache", "alldata", "corrector")


def learned(mod):
    spec = importlib.util.spec_from_file_location("alldata_" + mod, os.path.join(ROOT, "alldata", "learned", mod + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def one(args):
    path, base, version, key, sub = args
    name = os.path.basename(path)[:-4]
    dst = os.path.join(OUT, sub, name + ".npz")
    if os.path.exists(dst):
        return name
    from control import control_labels
    build = learned("build")
    d = dict(np.load(path))
    prov = name.split("_")[0]
    z = np.load(os.path.join(ROOT, "data", "cache", "alldata", "runs", base, prov, name + ".npz"))
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    fold = control_labels(d)
    bxy = build.ball_version(d, name, version)
    code = lambda l: np.select([l == "H", l == "A", l == "D"], [0, 1, 2], -1).astype(np.int8)
    out = {}
    for i, (a, b) in enumerate(halves):
        F, names, fr = cfeats.half_rows(d["players"], bxy, d["team"], a, b, d["pitch"], int(d["fps"]), z[f"{key}{i}"], z[f"f{key}{i}"])
        out[f"F_{i}"], out[f"y_{i}"], out[f"yf_{i}"], out[f"fr_{i}"] = F, code(d["labels"][fr]), code(fold[fr]), fr
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst + ".tmp.npz", names=np.array(names), nhalves=len(halves), **out)
    os.replace(dst + ".tmp.npz", dst)
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pw1e")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--key", default="s")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    sub = f"feats_{a.base}_v{a.version}"
    paths = [p for p in D.external_paths() if "soccertrack" not in p]
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, [(p, a.base, a.version, a.key, sub) for p in paths]):
            print(r, flush=True)
