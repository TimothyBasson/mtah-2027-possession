"""How much does a trained model care which tracking system the players came from?

    python deep/shift.py --tag v2 [--out data/cache/deep/shift_v2.json]

Takes each fold's net (deep/train.py), changes the held-out matches' tracking one way at a time
(deep/trackers.py: noise, smoothing, frame rate, grid, calibration, missing players), predicts
again and scores. Nothing is trained. A big drop under a mild change says the model leans on a
provider's tracking quirks, and that training on changed tracking (train.py --style) should help
on an unseen provider; no drop says the provider gap is somewhere else (the labels, the football).

Scored at one fixed decode against the labels as built and against the labels folded to the
controlled-possession rule (external/control.py) with a 2 s / 3 s minimum run.
"""

import argparse
import glob
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import data as D  # noqa: E402
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402
import trackers as TR  # noqa: E402
from build import speed  # noqa: E402
from control import control_labels  # noqa: E402
from net import PossessionNet  # noqa: E402
from train import predict_half  # noqa: E402

CHANGES = {
    "as recorded": [],
    "noise 0.1 m": [("white", dict(sigma=0.1))],
    "noise 0.3 m": [("white", dict(sigma=0.3))],
    "drift 0.5 m over 3 s": [("drift", dict(sigma=0.5, seconds=3))],
    "smoothed 0.4 s": [("smooth", dict(seconds=0.4))],
    "smoothed 1.2 s": [("smooth", dict(seconds=1.2))],
    "10 fps underneath": [("resample", dict(rate=10))],
    "5 fps underneath": [("resample", dict(rate=5))],
    "grid 1.05 x 0.68 m": [("grid", dict(gx=1.05, gy=0.68))],
    "pitch 4% too long": [("affine", dict(sx=1.04))],
    "shifted 1 m, 1 m": [("affine", dict(dx=1.0, dy=1.0))],
    "lost 7% in 5 s runs": [("lose", dict(share=0.07, seconds=5))],
    "lost 15% in 5 s runs": [("lose", dict(share=0.15, seconds=5))],
    "broadcast view 60 m": [("view", dict(width=60))],
}
RAW = dict(cost=4, min_team=1, min_dead=1)
FOLDED = dict(cost=4, min_team=2, min_dead=3)


def one(args):
    netpath, name, names = args
    torch.set_num_threads(1)
    d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    fps, pitch = int(d["fps"]), tuple(d["pitch"])
    net = PossessionNet(nfeat=D.NFEAT)
    net.load_state_dict(torch.load(netpath))
    truth = pm.label_codes(d["labels"])
    folded = pm.label_codes(control_labels(d))
    out = {}
    for label in names:
        rng = np.random.default_rng(zlib.crc32((label + name).encode()))
        players = d["players"].copy()
        probs = []
        for a, b in halves:
            if CHANGES[label]:
                players[a:b + 1, :, :2] = TR.apply(players[a:b + 1, :, :2], CHANGES[label], rng,
                                                   (pitch[0] / 2, pitch[1] / 2), fps)
                players[a:b + 1, :, 2] = speed(players[a:b + 1, :, :2], fps)
            X, M, bxy, _ = D.half_tensors(players, d["ball"], d["team"], a, b, pitch, fps)
            probs.append(predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy)))
        row = []
        for t, dec in ((truth, RAW), (folded, FOLDED)):
            r = pm.evaluate(t, pm.label_codes(decode.labels_from(probs, halves, len(t), **dec)), bootstrap_n=0)
            row += [r["spell_f1_mean"], r["alive_dead_f1"], r["team_acc_both_alive"]]
        out[label] = row
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--kind", default="test", help="test: the held-out provider; val: unseen matches of trained providers")
    ap.add_argument("--per", type=int, default=6, help="at most this many matches per provider and fold")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    names = [n for n in CHANGES if not a.only or n in a.only or n == "as recorded"]
    base = os.path.join(ROOT, "data", "cache", "deep", a.tag)
    jobs = []
    for fold in sorted(os.listdir(base)):
        if not os.path.exists(os.path.join(base, fold, "net.pt")):
            continue
        seen = {}
        for f in sorted(glob.glob(os.path.join(base, fold, "*.npz"))):
            if str(np.load(f)["kind"]) != a.kind:
                continue
            name = os.path.basename(f)[:-4]
            seen.setdefault(D.provider(f), []).append(name)
        for prov, ms in seen.items():
            jobs += [(os.path.join(base, fold, "net.pt"), m, names) for m in ms[:a.per]]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one, jobs))
    provs = sorted({n.split("_")[0] for n in res})
    print(f"{'change':24s}" + "".join(f"{p[:8]:>9s}" for p in provs) + "     mean |" +
          "".join(f"{p[:8]:>9s}" for p in provs) + "     mean   (spellF1: labels as built | folded)")
    table = {}
    for label in names:
        cells = []
        for k in (0, 3):
            per = [np.mean([res[n][label][k] for n in res if n.startswith(p)]) for p in provs]
            cells.append(per + [np.mean(per)])
        table[label] = cells
        print(f"{label:24s}" + " |".join("".join(f"{x:9.3f}" for x in c) for c in cells))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(table=table, providers=provs, matches=res), f)


if __name__ == "__main__":
    main()
