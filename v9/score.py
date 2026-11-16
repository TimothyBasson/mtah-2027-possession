"""Held-out spellF1 of v9 candidates against hmm_v8, on the yardstick of external/score_control.py.

    python v9/score.py --nets sq,sqs1 [--nets2 bt] [--wd 0.1,0.2,0.3] [--wc 0.2,0.1,0]

Each match is decoded by hmm/'s decoder with its v6 clues, plus wc x the control classifier's
log-ratio (hmm_v8 is wc 0.2, wd 0) plus wd x the sequence model's log-probabilities, averaged over
the nets named (v9/train.py fold nets: no match is scored by a net or a classifier that saw its
provider). "seq alone" rows decode the net's probabilities with deep/'s own decode. Raw and folded
labels as score_control.py. The last lines pick (wc, wd) for each provider on the other two.
"""

import argparse
import glob
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external"), os.path.join(ROOT, "deep")]
import hmm.decode as dec  # noqa: E402
import score_hmm as sh  # noqa: E402
import score_control as sc  # noqa: E402
import decode as dd  # noqa: E402
from control import control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402

STATE = {}


def _init(state):
    STATE.update(state)


def net_logp(name, tags, d):
    """(frames, 3) mean log-probability of the nets at 25 fps on the halves, 0 elsewhere."""
    k = name.split("_")[0]
    out = np.zeros((len(d["labels"]), 3))
    zs = [np.load(os.path.join(ROOT, "data", "cache", t.split(":")[0], t.split(":")[1], k, name + ".npz"))
          for t in tags]
    for i, (a, b) in enumerate(d["halves"]):
        a, b = int(a), int(b)
        p = np.mean([dd.upsample(z[f"p{i}"], b - a + 1) for z in zs], 0)
        out[a:b + 1] = np.log(np.clip(p, 1e-4, 1))
    return out


def run(name):
    paths, tabs, loso = (STATE[k] for k in ("paths", "tables", "loso"))
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = tabs[k]
    d = sh.load(paths[name])
    ball = match_ball(name, d)
    views = (d["labels"], control_labels(d))
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t),
             (sh.flight_cats(d, ball), flight_t)]
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - np.log(loso["prior_" + name])
    trans, start = dec.transitions(), np.log(dec.START)
    out = {}

    def keep(key, pred):
        pred[d["labels"] == "X"] = "X"
        out[key] = [sh.score(gt, pred)[0] for gt in views]

    for si, tags in enumerate(STATE["sets"]):
        lp = net_logp(name, tags, d)
        prior = np.log(STATE["prior"])
        for wc, wd, hw in STATE["grid"]:
            if si and wd == 0:
                continue
            direct = wc * ratio + wd * (lp - prior)
            keep((si, wc, wd, hw), dec.decode(clues, trans, start, segments, kickoffs=(), extra=extra,
                                              direct=direct, weight=hw * dec.WEIGHT if hw else 0.0))
        probs = [np.exp(lp[a:b][::5]) for a, b in segments]
        spans = [(a, b - 1) for a, b in segments]
        for c in STATE["alone"]:
            keep((si, "alone") + tuple(sorted(c.items())), dd.labels_from(probs, spans, len(lp), **c))
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nets", action="append", required=True,
                    help="comma-separated folder:tag (v9:sq, deep:bt) averaged; repeat for another set")
    ap.add_argument("--wd", default="0,0.1,0.2,0.3,0.5")
    ap.add_argument("--wc", default="0.2,0.1,0")
    ap.add_argument("--hw", default="1", help="multipliers of the v6 clues' weight (0: none)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    grid = list(itertools.product(*[[float(v) for v in s.split(",")] for s in (a.wc, a.wd, a.hw)]))
    alone = [dict(cost=4, min_team=1, min_dead=1), dict(cost=4, min_team=2, min_dead=1),
             dict(cost=8, min_team=2, min_dead=1)]
    state = dict(paths=paths, tables=sc.tables(paths), loso=dict(np.load(os.path.join(CACHE, "control_loso.npz"))),
                 sets=[s.split(",") for s in a.nets], grid=grid, alone=alone, prior=np.array([0.3, 0.3, 0.4]))
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        rows = dict(pool.map(run, list(paths)))
    keys = list(next(iter(rows.values())))
    table = {}
    for key in keys:
        table[key] = np.array([[np.mean([rows[n][key][v] for n in rows if n.startswith(k)]) for k in sh.SETS]
                               for v in (0, 1)])
    print("set  wc   wd   hw  | raw: idsse metrica skillc mean | folded: idsse metrica skillc mean")
    for key in sorted(keys, key=str):
        t = table[key]
        print("{:42s} | {} {:.3f} | {} {:.3f}".format(
            str(key), " ".join("{:.3f}".format(x) for x in t[0]), t[0].mean(),
            " ".join("{:.3f}".format(x) for x in t[1]), t[1].mean()))
    for si in range(len(a.nets)):
        cand = [k for k in keys if k[0] == si and k[1] != "alone"]
        fair = []
        for i in range(3):
            others = [j for j in range(3) if j != i]
            pick = max(cand, key=lambda k: table[k][1][others].mean())
            fair.append((table[pick][1][i], pick[1:]))
        print("set {} fair (settings picked on the other two providers, folded): {} mean {:.3f}  picks {}".format(
            si, " ".join("{:.3f}".format(f[0]) for f in fair), np.mean([f[0] for f in fair]), [f[1] for f in fair]))


if __name__ == "__main__":
    main()
