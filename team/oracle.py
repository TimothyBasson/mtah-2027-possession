"""What a perfect team-naming step would be worth on top of stage 1, on folded labels.

    python team/oracle.py [--tags v2,v2s1]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import decode  # noqa: E402


def maj(gt, a, b, fallback):
    h, aw = np.sum(gt[a:b] == "H"), np.sum(gt[a:b] == "A")
    if h + aw == 0:
        return fallback
    return "H" if h >= aw else "A"


def one(args):
    name, tags = args
    raw, fold, spans = C.truth(name)
    probs = np.mean([np.concatenate(C.stage1(t, name)) for t in tags], 0)
    cut = np.cumsum([0] + [len(p) for p in C.stage1(tags[0], name)])
    probs = [probs[cut[i]:cut[i + 1]] for i in range(len(spans))]
    pred = decode.labels_from(probs, spans, len(raw), **C.DECODE)
    out = {"base": pred}
    # B: stage-1 possessions, each named by the true majority team
    b = pred.copy()
    stats = []
    for s, e, v in C.runs(pred):
        if v in ("H", "A"):
            t = maj(fold, s, e, v)
            b[s:e] = t
            live = np.isin(fold[s:e], ["H", "A"])
            pure = np.mean(fold[s:e][live] == t) if live.any() else np.nan
            stats.append(((e - s) / 25, v == t, pure, live.mean()))
    out["segs_true_team"] = b
    # D: each stage-1 live stretch one team (no turnovers inside)
    dd = pred.copy()
    live = pred != "D"
    for s, e, v in C.runs(live & (pred != "X")):
        if v:
            dd[s:e] = maj(fold, s, e, "H")
    out["stretch_one_team"] = dd
    # C: stage-1 alive/dead, true team frame by frame (true-dead frames take the nearest true team)
    c = pred.copy()
    idx = np.flatnonzero(np.isin(fold, ["H", "A"]))
    near = idx[np.clip(np.searchsorted(idx, np.arange(len(fold))), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(len(fold)), side="right") - 1, 0, len(idx) - 1)]
    use = np.where(np.abs(prev - np.arange(len(fold))) <= np.abs(near - np.arange(len(fold))), prev, near)
    lm = np.isin(pred, ["H", "A"])
    c[lm] = fold[use][lm]
    out["frames_true_team"] = c
    # E: true alive/dead and true turnover times, stage-1 votes the team of each true possession
    e_ = fold.copy()
    p25 = np.zeros((len(raw), 3))
    for p, (a, bb) in zip(probs, spans):
        p25[a:bb + 1] = decode.upsample(p, bb - a + 1)
    for s, e, v in C.runs(fold):
        if v in ("H", "A"):
            l = np.log(np.clip(p25[s:e], 1e-6, 1)).sum(0)
            e_[s:e] = "H" if l[0] >= l[1] else "A"
    out["true_bounds_model_vote"] = e_
    sc = {k: (C.spell_f1(fold, v), C.spell_f1(raw, v)) for k, v in out.items()}
    return name, sc, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v2")
    a = ap.parse_args()
    tags = a.tags.split(",")
    names = [os.path.basename(p)[:-4] for p in C.paths()]
    with Pool(4) as pool:
        res = pool.map(one, [(n, tags) for n in names])
    keys = list(res[0][1])
    print(f"{'folded-label spellF1':26s} idsse  metrica  skillc   mean   (raw-label mean)")
    for k in keys:
        f = C.provider_mean(names, [r[1][k][0] for r in res])
        rw = C.provider_mean(names, [r[1][k][1] for r in res])
        print(f"{k:26s} " + "  ".join(f"{x:.3f}" for x in f) + f"   ({rw[-1]:.3f})")
    for prov in C.PROVIDERS:
        st = np.array([s for n, _, ss in res if n.startswith(prov) for s in ss])
        print(f"\n{prov}: stage-1 possessions by length: share, team right, purity when named right, share of frames truly live")
        for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)):
            m = (st[:, 0] >= lo) & (st[:, 0] < hi)
            if m.any():
                print(f"  {lo:>3}-{hi if hi < 1e8 else 'inf':>4} s  n={m.sum():5d} {m.mean():.2f}  right {st[m, 1].mean():.2f}  "
                      f"purity {np.nanmean(st[m, 2]):.2f}  live {st[m, 3].mean():.2f}")


if __name__ == "__main__":
    main()
