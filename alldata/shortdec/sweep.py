"""Score decode settings on the saved held-out run: spellF1 on labels as built and folded, and
the merge slice. Settings are picked on each fold's validation matches (training providers the
fold's nets did not train on) and scored on the held-out provider.

    python alldata/shortdec/sweep.py --tag pw1e --grid basic
"""
import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import dec  # noqa: E402
from common import pm  # noqa: E402

BASE = dict(cost=4.0, min_team=1.0, min_dead=1.0)


def grid(which):
    if which == "basic":
        return [BASE] + [dict(cost=c, ha=h, min_team=t, min_dead=d, power=pw)
                         for c in (2, 4, 6) for h in (None, c / 2, c * 2) for t in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
                         for d in (1.0, 3.0) for pw in (1.0,)]
    if which == "sand":
        return [BASE] + [dict(cost=c, min_team=t, min_dead=d, min_sand=s, sand_p=sp)
                         for c in (2, 4) for t in (1.0, 1.5, 2.0) for d in (1.0, 3.0)
                         for s in (0.2, 0.4, 0.6, 0.8) for sp in (0.0, 0.5, 0.7, 0.85)]
    raise SystemExit(which)


def one(args):
    fold, name, f, key, combos = args
    probs, spans, raw, fd = C.load(f, key)
    Ps = [dec.upsample(p, b - a + 1) for p, (a, b) in zip(probs, spans)]
    truths = [(t, pm.valid_runs(t)) for t in (raw, fd)]
    out = []
    for c in combos:
        pred = dec.decode(Ps, spans, len(raw), **c)
        row = []
        for t, runs in truths:
            f1, st = C.slice_stats(t, pm.fill_abstentions(pred, runs), runs)
            row.append(np.r_[f1, st])
        out.append(row)
    return np.array(out)  # (combo, labelling, 8)


def run(tag, combos, key="s", kind="test"):
    ms = C.matches(tag, kind)
    with ProcessPoolExecutor(4) as ex:
        R = np.array(list(ex.map(one, [(fo, n, f, key, combos) for fo, n, f in ms])))
    return ms, R  # (match, combo, labelling, 8)


def fmt(ms, R, ci):
    names = [n for _, n, _ in ms]
    out = []
    for li in range(2):
        f = C.provider_mean(names, R[:, ci, li, 0])
        s = C.provider_sum(names, R[:, ci, li, 1:]).sum(0)
        out.append(" ".join(f"{x:.3f}" for x in f) + f" | brief {s[1] / s[0]:.2f} nb {s[3] / s[2]:.2f} merged {s[4] / s[0]:.2f} drawn/true {s[6] / s[5]:.2f}")
    return "   ||   ".join(out)


def val_pick(tag, combos, key, weights=(0.5, 0.5)):
    """Per fold: the setting with the best validation score (providers equal; mean of the two
    labellings), and that setting's test rows."""
    msv, Rv = run(tag, combos, key, "val")
    picks = {}
    for fold in C.PROVS:
        idx = [i for i, (fo, n, _) in enumerate(msv) if fo == fold]
        provs = sorted({msv[i][1].split("_")[0] for i in idx})
        v = np.mean([np.mean([Rv[i, :, :, 0] for i in idx if msv[i][1].startswith(p)], 0) for p in provs], 0)
        picks[fold] = int((v @ np.array(weights)).argmax())
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--key", default="s")
    ap.add_argument("--grid", default="basic")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    combos = grid(a.grid)
    ms, R = run(a.tag, combos, a.key)
    names = [n for _, n, _ in ms]
    print("columns: IDSSE Metrica SkillCorner mean, labels as built || folded")
    print("shipped  ", fmt(ms, R, 0))
    tot = np.array([[C.provider_mean(names, R[:, ci, li, 0])[3] for li in range(2)] for ci in range(len(combos))])
    for ci in np.argsort(-tot.sum(1))[:a.top]:
        print(fmt(ms, R, ci), combos[ci])
    picks = val_pick(a.tag, combos, a.key)
    per = np.zeros((2, 4)); 
    for li in range(2):
        for pi, pv in enumerate(C.PROVS):
            per[li, pi] = np.mean([R[i, picks[pv], li, 0] for i, n in enumerate(names) if n.startswith(pv)])
        per[li, 3] = per[li, :3].mean()
    print("picked on validation:", " || ".join(" ".join(f"{x:.3f}" for x in per[li]) for li in range(2)))
    for pv in C.PROVS:
        print("  ", pv, combos[picks[pv]])


if __name__ == "__main__":
    main()
