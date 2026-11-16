"""What would a turnover detector be worth? The held-out bt probabilities decoded with the true
turnover times (and with degraded copies of them) as evidence for an H<->A switch.

    python turnover/oracle.py
"""
import sys
from multiprocessing import Pool

import numpy as np

import common as C
from common import sh

TAGS = ["bt", "bts1"]


def events(frames, n, rng, recall=1.0, fp_per_min=0.0, jitter=0.0, live=None):
    f = frames[rng.random(len(frames)) < recall]
    if jitter:
        f = np.clip(f + np.round(rng.normal(0, jitter * 25, len(f))).astype(int), 1, n - 1)
    if fp_per_min:
        cand = np.flatnonzero(live)
        f = np.r_[f, rng.choice(cand, int(len(cand) / 1500 * fp_per_min))]
    return f


def bonus_from(frames, n, size, width):
    b = np.zeros(n)
    for k in range(-width, width + 1):
        i = frames + k
        b[i[(i > 0) & (i < n)]] = size
    return b


CONDS = [("base", None),
         ("bonus 8", dict(size=8)),
         ("bonus 16", dict(size=16)), ("bonus 40", dict(size=40)),
         ("bonus 28, H<->A cost 16 elsewhere", dict(size=28, ha=16)),
         ("bonus 60, H<->A cost 40 elsewhere", dict(size=60, ha=40)),
         ("same, min possession 0.4 s", dict(size=60, ha=40, mt=0.4)),
         ("same, recall 0.9", dict(size=60, ha=40, recall=0.9)),
         ("same, recall 0.7", dict(size=60, ha=40, recall=0.7)),
         ("bonus 28/16, recall 0.7", dict(size=28, ha=16, recall=0.7)),
         ("bonus 28/16, recall 0.7 + 2 false/min", dict(size=28, ha=16, recall=0.7, fp_per_min=2)),
         ("bonus 16, recall 0.7 + 2 false/min", dict(size=16, recall=0.7, fp_per_min=2)),
         ("bonus 16, recall 0.7 + 2 false/min, jitter .2", dict(size=16, recall=0.7, fp_per_min=2, jitter=0.2,width=6)),
         ]


def one(path):
    name = path.split("/")[-1][:-4]
    d = sh.load(path)
    spans = [tuple(h) for h in d["halves"]]
    n = len(d["labels"])
    probs = C.base_probs(name, TAGS, len(spans))
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    out = []
    for gt in (d["labels"], C.control_labels(d)):
        tv = C.turnovers(gt)
        live = (gt == "H") | (gt == "A")
        row = []
        for _, c in CONDS:
            ha, mt = None, 1
            if c is None:
                bonus = None
            else:
                c = dict(c)
                f = tv
                if "short" in c:
                    s = c.pop("short") * 25
                    ch = np.r_[0, np.flatnonzero(gt[1:] != gt[:-1]) + 1, n]
                    keep = []
                    for t in tv:  # the possession starting or ending at t is short
                        k = np.searchsorted(ch, t)
                        if ch[k + 1] - ch[k] < s or ch[k] - ch[k - 1] < s:
                            keep.append(t)
                    f = np.array(keep, int)
                size, width = c.pop("size"), c.pop("width", 3)
                ha, mt = c.pop('ha', None), c.pop('mt', 1)
                bonus = bonus_from(events(f, n, rng, live=live, **c), n, size, width)
            pred = C.labels_ev(probs, spans, n, bonus, ha_cost=ha, min_team=mt)
            pred[d["labels"] == "X"] = "X"
            row.append(sh.score(gt, pred)[0])
        out.append(row)
    return name, np.array(out), len(C.turnovers(d["labels"])), len(C.turnovers(C.control_labels(d)))


if __name__ == "__main__":
    with Pool(4) as pool:
        rows = pool.map(one, C.paths())
    names = [r[0] for r in rows]
    S = np.stack([r[1] for r in rows])
    for ti, tn in enumerate(("labels as given", "folded labels")):
        print(f"\n{tn}:  idsse metrica skillcorner mean")
        for ci, (cn, _) in enumerate(CONDS):
            print(f"  {cn:42s}" + "  ".join(f"{x:.3f}" for x in C.provider_mean(names, S[:, ti, ci])))
    print("turnovers per match raw/folded", C.provider_mean(names, [[r[2], r[3]] for r in rows]).round(0))
