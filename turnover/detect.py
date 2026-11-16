"""The detector as a turnover finder against the base model's own turnovers, held out.

    python turnover/detect.py --det t1

Detector peaks (a local maximum of "to home" or "to away" above a threshold, one per second)
against the true direct H<->A changes: found when a peak of the same direction lies within 1 s.
Also how many of the turnovers the base model misses the detector finds, and the detector's own
H/A/D side head decoded alone and averaged with the base.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from common import sh  # noqa: E402
from recall import match_events  # noqa: E402

THR = (0.1, 0.2, 0.3, 0.5)


def peaks(p, a, thr, gap=12):
    """[(frame at 25 fps, 'H' or 'A')] from (T, 3) at 12.5 fps."""
    out = []
    for k, letter in ((1, "H"), (2, "A")):
        q = p[:, k].astype(float)
        idx = np.flatnonzero((q > thr) & (q >= np.r_[q[1:], 0]) & (q > np.r_[0, q[:-1]]))
        keep = []
        for i in idx[np.argsort(-q[idx])]:
            if all(abs(i - j) > gap for j in keep):
                keep.append(i)
        out += [(a + 2 * i, letter) for i in keep]
    out.sort()
    return np.array([f for f, _ in out], int), np.array([l for _, l in out])


def one(args):
    fold, name, base, det, key = args
    d = sh.load(os.path.join(C.ROOT, "data", "external", name + ".npz"))
    spans = [tuple(h) for h in d["halves"]]
    n = len(d["labels"])
    probs = C.base_probs(name, base, len(spans)) if fold == name.split("_")[0] else None
    z = np.load(os.path.join(C.ROOT, "data", "cache", "turnover", det, fold, name + ".npz"))
    gt = d["labels"] if key == "ev" else C.control_labels(d)
    tv = C.turnovers(gt)
    pred = C.labels_ev(probs, spans, n)
    pred[d["labels"] == "X"] = "X"
    bf = C.turnovers(pred)
    hb, pb = match_events(tv, gt[tv], bf, pred[bf], 25)
    row = [[hb.sum(), len(tv), pb.sum(), len(bf), 0, 0]]
    for thr in THR:
        pk = [peaks(z[f"{key}{i}"], a, thr) for i, (a, b) in enumerate(spans)]
        f, l = np.concatenate([x[0] for x in pk]), np.concatenate([x[1] for x in pk])
        live = np.isin(gt[np.clip(f, 0, n - 1)], ("H", "A"))
        f, l = f[live], l[live]
        ht, hp = match_events(tv, gt[tv], f, l, 25)
        row.append([ht.sum(), len(tv), hp.sum(), len(f), (ht & ~hb).sum(), (~hb).sum()])
    # the side head as a possession model of its own, and averaged with the base
    cls = [C.decode.upsample(np.repeat(z[f"cls{i}"].astype(float), 1, 0)[::1], 1) if False else z[f"cls{i}"].astype(float) for i in range(len(spans))]
    sc = []
    for mix in (0, 1, 0.5):
        pr = []
        for i, (a, b) in enumerate(spans):
            t = np.arange(len(cls[i])) * 2
            c25 = np.stack([np.interp(np.arange(b - a + 1), t, cls[i][:, k]) for k in range(3)], 1)
            b25 = C.decode.upsample(probs[i], b - a + 1)
            pr.append((mix * c25 + (1 - mix) * b25)[::5])
        p2 = C.labels_ev(pr, spans, n)
        p2[d["labels"] == "X"] = "X"
        sc.append([sh.score(g, p2)[0] for g in (d["labels"], C.control_labels(d))])
    return name, np.array(row), np.array(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="t1")
    ap.add_argument("--base", default="bt,bts1")
    ap.add_argument("--key", default="ev")
    a = ap.parse_args()
    jobs = [(os.path.basename(p).split("_")[0], os.path.basename(p)[:-4], a.base.split(","), a.det, a.key) for p in C.paths()
            if os.path.exists(os.path.join(C.ROOT, "data", "cache", "turnover", a.det, os.path.basename(p).split("_")[0], os.path.basename(p)))]
    with Pool(4) as pool:
        res = pool.map(one, jobs)
    for p in C.PROVIDERS:
        R = [r for n, r, s in res if n.startswith(p)]
        if not R:
            continue
        R = np.sum(R, 0); S = np.mean([s for n, r, s in res if n.startswith(p)], 0)
        print(f"\n{p}: {R[0, 1] / len([1 for n, _, _ in res if n.startswith(p)]):.0f} true turnovers a match ({a.key})")
        print(f"  base model's own turnovers: recall {R[0, 0] / R[0, 1]:.2f} precision {R[0, 2] / R[0, 3]:.2f}")
        for t, r in zip(THR, R[1:]):
            print(f"  detector above {t}: recall {r[0] / r[1]:.2f} precision {r[2] / r[3]:.2f}; of those the base misses it finds {r[4] / r[5]:.2f}")
        print("  spellF1 raw/folded: base %.3f/%.3f  detector's H/A/D head alone %.3f/%.3f  averaged %.3f/%.3f" % tuple(S.ravel()))


if __name__ == "__main__":
    main()
