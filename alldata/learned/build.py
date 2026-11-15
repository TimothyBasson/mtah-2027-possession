"""Feature cache for the all-data model: data/cache/alldata/feats/<match>.npz.

    python alldata/learned/build.py [--versions 3] [--jobs 4]

Per match and half: the feature rows with the ball as recorded (v0) and with BAR-ATH-like
visibility simulated under `versions` seeds (v1..), the labels as built and folded to the
controlled-possession rule (external/control.py), and the frame of each row.

The players-only net's probabilities (bt, bts1) enter as inputs. Each match takes them from the
fold in which its provider was held out, so for every match they are what an unseen provider
gets, which is BAR-ATH's situation.
"""

import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), ROOT]
import data as D  # noqa: E402
import feats  # noqa: E402
import tim  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "alldata", "feats")
BT = ("bt",)  # one net, as on BAR-ATH (deep/models/bt.pt)


def bt_probs(name, nhalves):
    prov = name.split("_")[0]
    out = []
    for tag in BT:
        f = os.path.join(ROOT, "data", "cache", "deep", tag, prov, name + ".npz")
        if not os.path.exists(f):
            return None
        z = np.load(f)
        out.append([z[f"p{i}"] for i in range(nhalves)])
    return [np.mean([o[i] for o in out], 0) for i in range(nhalves)]


def one(args):
    path, versions = args
    name = os.path.basename(path)[:-4]
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        return name
    from control import control_labels
    d = dict(np.load(path))
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    lab = d["labels"]
    fold = control_labels(d)
    bt = bt_probs(name, len(halves))
    if bt is None:
        return name + " (no bt probabilities, skipped)"
    # Tim's HMM as optional inputs (tim.py); fit.py --drop-cols tim trains without them
    tim_in = tim.inputs(name, [np.arange(a, b + 1, feats.STEP) for a, b in halves])
    code = lambda l: np.select([l == "H", l == "A", l == "D"], [0, 1, 2], -1).astype(np.int8)  # noqa: E731
    out = {}
    for v in range(versions + 1):
        rng = np.random.default_rng(zlib.crc32(f"{name} {v}".encode()))
        bxy = d["ball"][:, :2].astype(np.float32).copy()
        if v:
            for a, b in halves:
                bxy[a:b + 1] = feats.simulate_visibility(bxy[a:b + 1], lab[a:b + 1], rng, int(d["fps"]))
        for i, (a, b) in enumerate(halves):
            F, fr = feats.half_features(d["players"], bxy, d["team"], a, b, d["pitch"], int(d["fps"]),
                                        extra=[("bt", bt[i])] + tim_in[i])
            out[f"F{v}_{i}"] = F
            if v == 0:
                out[f"y_{i}"], out[f"yf_{i}"], out[f"fr_{i}"] = code(lab[fr]), code(fold[fr]), fr
    os.makedirs(OUT, exist_ok=True)
    np.savez(dst + ".tmp.npz", names=np.array(feats.NAMES), nhalves=len(halves), **out)
    os.replace(dst + ".tmp.npz", dst)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    paths = D.external_paths()
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, [(p, a.versions) for p in paths]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
