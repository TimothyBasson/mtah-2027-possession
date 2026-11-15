"""Where the v8 decoder loses spellF1 against the control-folded labels: F1 at each IoU, spell
counts, and oracles that fix one kind of error at a time with the truth (not usable, only to size
the headroom of each kind).

    python external/post_diag.py
"""

import numpy as np

import post_lib as pl
import possession_metrics as pm
from post_lib import A, D, H

NAMES = np.array([pm.H, pm.A, pm.D])


def runs_of(path):
    edges = np.flatnonzero(path[1:] != path[:-1]) + 1
    return np.r_[0, edges], np.r_[edges, len(path)]


def gt_path(m):
    g = m["gt"]
    return np.select([g == pm.H, g == pm.A, g == pm.D], [H, A, D], -1).astype(np.int8)


def oracle_dead(m, emit, trans):
    """Truth's alive/dead, the decoder's team: Viterbi with D forced where the truth is D and
    ruled out where it is alive."""
    g = gt_path(m)
    e = emit.copy()
    e[g == D, :2] = -1e9
    e[(g == H) | (g == A), 2] = -1e9
    return pl.decode(m, e, trans)


def oracle_team(m, path):
    """The decoder's alive/dead, the truth's team where both are alive."""
    g = gt_path(m)
    out = path.copy()
    both = (path != D) & (path >= 0) & ((g == H) | (g == A))
    out[both] = g[both]
    return out


def oracle_snap(m, path, seconds):
    """Every predicted change moved to the nearest true change of the same kind within seconds."""
    g = gt_path(m)
    fps = int(m["fps"])
    out = path.copy()
    gt_edges = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts, stops = runs_of(path)
    for i in range(1, len(starts)):
        e = starts[i]
        a, b = path[e - 1], path[e]
        if a < 0 or b < 0:
            continue
        cand = gt_edges[(np.abs(gt_edges - e) <= seconds * fps)]
        cand = cand[(g[cand - 1] == a) & (g[cand] == b)]
        if not len(cand):
            continue
        t = cand[np.argmin(np.abs(cand - e))]
        lo = starts[i - 1] + 1
        hi = stops[i] - 1
        t = int(np.clip(t, lo, hi))
        if t < e:
            out[t:e] = b
        else:
            out[e:t] = a
    return out


def main():
    ms = pl.load_all()
    rows = {}
    for n, m in ms.items():
        emit, trans = pl.emissions(m), pl.transitions(int(m["fps"]))
        path = pl.decode(m, emit, trans)
        s = pl.spell(m, path)
        r = {"v8": s["mean_f1"], "v8 raw labels": pl.spell(m, path, "gt_raw")["mean_f1"]}
        for t in pm.SPELL_IOUS:
            r["F1@{}".format(t)] = s["per_iou"][t]["f1"]
        r["P@0.5"], r["R@0.5"] = s["per_iou"][0.5]["precision"], s["per_iou"][0.5]["recall"]
        r["n_pred/n_gt"] = s["n_pred"] / s["n_gt"]
        r["oracle: true alive/dead"] = pl.spell(m, oracle_dead(m, emit, trans))["mean_f1"]
        r["oracle: true team"] = pl.spell(m, oracle_team(m, path))["mean_f1"]
        for sec in (0.5, 1, 2, 4):
            r["oracle: snap {} s".format(sec)] = pl.spell(m, oracle_snap(m, path, sec))["mean_f1"]
        rows[n] = r
    for key in next(iter(rows.values())):
        print("{:26s} {}".format(key, pl.fmt(pl.by_set({n: r[key] for n, r in rows.items()}))))


if __name__ == "__main__":
    main()
