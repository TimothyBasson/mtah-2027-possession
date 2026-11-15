"""What the detector's false alarms are. For each held-out peak with no true turnover of its
direction within 1 s: what the labels and the recorded ball did around it (analysis only).

    python turnover/alarms.py --det t1
"""
import argparse
import os
import sys
import warnings
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from common import sh  # noqa: E402
from detect import peaks  # noqa: E402
from recall import match_events  # noqa: E402
from hmm.ball import detect_touches  # noqa: E402


def one(name, det, thr):
    fold = name.split("_")[0]
    d = sh.load(os.path.join(C.ROOT, "data", "external", name + ".npz"))
    z = np.load(os.path.join(C.ROOT, "data", "cache", "turnover", det, fold, name + ".npz"))
    gt, n = d["labels"], len(d["labels"])
    spans = [tuple(h) for h in d["halves"]]
    pk = [peaks(z[f"ev{i}"], a, thr) for i, (a, b) in enumerate(spans)]
    f, l = np.concatenate([x[0] for x in pk]), np.concatenate([x[1] for x in pk])
    tv = C.turnovers(gt)
    _, hp = match_events(tv, gt[tv], f, l, 25)
    ball, pl = d["ball"].astype(float), d["players"].astype(float)
    dist = np.hypot(pl[:, :, 0] - ball[:, None, 0], pl[:, :, 1] - ball[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dh, da = np.nanmin(dist[:, d["team"] == 0], 1), np.nanmin(dist[:, d["team"] == 1], 1)
    t = detect_touches(ball[:, 0], ball[:, 1], dh, da, 25)
    c = Counter()
    for fr, to, hit in zip(f, l, hp):
        fr = min(fr, n - 1)
        if hit:
            c["a true turnover within 1 s"] += 1
            continue
        w = gt[max(0, fr - 25):fr + 26]
        near = tv[np.abs(tv - fr) <= 50]
        tt = t["team"][np.abs(t["frame"] - fr) <= 12]
        me = 0 if to == "H" else 1
        if gt[fr] in "DX" or "D" in w or "X" in w:
            c["at a stoppage or restart (dead within 1 s)"] += 1
        elif len(near) and (gt[near] == to).any():
            c["a true turnover 1 to 2 s away"] += 1
        elif (np.abs(tv - fr) <= 25).any():
            c["a true turnover within 1 s, the other way"] += 1
        elif gt[fr] == to and (tt == me).any():
            c["the named team already has it and touches the ball (a pass or carry)"] += 1
        elif gt[fr] != to and (tt == me).any():
            c["the named team touches the ball but the label does not change"] += 1
        elif gt[fr] == to:
            c["the named team already has it, no touch"] += 1
        else:
            c["the other team has it, the named team does not touch it"] += 1
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="t1")
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    for p in C.PROVIDERS:
        names = [os.path.basename(x)[:-4] for x in C.paths() if os.path.basename(x).startswith(p)][:7]
        c = sum((one(nm, a.det, a.thr) for nm in names), Counter())
        tot = sum(c.values())
        print(f"\n{p}: {tot / len(names):.0f} peaks a match above {a.thr}")
        for k, v in c.most_common():
            print(f"  {100 * v / tot:4.0f}%  {k}")
