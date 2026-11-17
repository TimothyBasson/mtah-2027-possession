"""Held-out score of v9 + the keeper detector clue (external/score_fuse.py's yardstick, hold on).

    python keeper/score.py --variants "0:0.3:all,1:0.3:all,1:0.3:blind" [--pred keeper_pred]

Each variant is "keeper weight:net weight:gate" with gate all (every gated row) or blind (gated
rows with the ball unseen now). Control weight 0.2 and nets bt,b2 as v9."""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
import hmm.decode as dec  # noqa
import score_control as sc  # noqa
import score_hmm as sh  # noqa
import score_fuse as sf  # noqa
from control import CONTROL_SECONDS, control_labels  # noqa
from control_features import CACHE, match_ball  # noqa
from hmm.net_clue import log_ratio  # noqa
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa
from keeper.clue import keeper_direct  # noqa

STATE = {}


def _init(s):
    STATE.update(s)


def run(name):
    paths, tabs, loso, variants = (STATE[k] for k in ("paths", "tables", "loso", "variants"))
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = tabs[k]
    d = sh.load(paths[name])
    ball = match_ball(name, d)
    views = (d["labels"], control_labels(d, CONTROL_SECONDS))
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t),
             (sh.flight_cats(d, ball), flight_t)]
    width = sh.width_cats(d)
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - np.log(loso["prior_" + name])
    prior = np.mean([STATE["shares"][o] for o in sh.SETS if o != k], axis=0)
    net = log_ratio(sf.net_probs(name, len(d["halves"]), ["bt", "b2"]), d["halves"],
                    len(d["labels"]), prior)
    kz = STATE["kfeat"][name]
    kp = STATE["kpred"]
    seen_col = list(kz["names"]).index("seen")
    blind = kz["score_X"][:, seen_col] < 0.5
    kd = {g: keeper_direct(kp[name + "|score|p"], kp[name + "|prior"], kz["frames"], kz["ends"],
                           kz["score_gate"], kz["dteam"], len(d["labels"]),
                           only=None if g == "all" else blind) for g in ("all", "blind")}
    trans, start = dec.transitions(), np.log(dec.START)
    out = {}
    for label, (wk, wn, g) in variants.items():
        direct = 0.2 * ratio + wn * net + wk * kd[g]
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), weight=dec.WEIGHT,
                          extra=[(width, width_t, dec.WIDTH_WEIGHT)], direct=direct,
                          min_possession=0)
        pred = dec.hold_restarts(pred, bx, by, fps)
        pred[d["labels"] == "X"] = "X"
        out[label] = [sh.score(gt, pred) for gt in views]
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="0:0.3:all,0.5:0.3:all,1:0.3:all,0.5:0.3:blind,1:0.3:blind")
    ap.add_argument("--pred", default="keeper_pred")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    variants = {}
    for item in a.variants.split(","):
        wk, wn, g = item.split(":")
        variants[item] = (float(wk), float(wn), g)
    kfeat = {n: dict(np.load(os.path.join(ROOT, "data", "cache", "keeper", n + ".npz"))) for n in paths}
    state = dict(paths=paths, tables=sc.tables(paths), shares=sf.label_shares(paths),
                 loso=dict(np.load(os.path.join(CACHE, "control_loso.npz"))), variants=variants,
                 kfeat=kfeat, kpred=dict(np.load(os.path.join(ROOT, "data", "cache", a.pred + ".npz"))))
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        rows = dict(pool.map(run, list(paths), chunksize=1))
    sf.report(rows, next(iter(variants)))
    if a.out:
        np.savez(os.path.join(ROOT, "data", "cache", "kscore_{}.npz".format(a.out)),
                 names=np.array(list(rows)), variants=np.array(list(variants)),
                 scores=np.array([[rows[n][m] for m in variants] for n in rows]))


if __name__ == "__main__":
    main()
