"""hmm_v9 (restart hold on) plus the keeper-area ball clue (hmm/keeper.py), scored held out by
provider as external/score_fuse.py scores v9.

    python kp/score_keeper.py --variants "0,next:01:4,both:01:4,both:0123:4" [--out NAME]

A variant is ENDS:CATS:MULT (MULT times the decoder weight), or 0 for v9. Each provider is decoded
with a keeper table counted on the other two providers' labels only.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "external"))
import hmm.decode as dec  # noqa: E402
import score_control as sc  # noqa: E402
import score_fuse as sf  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm import ahead, keeper  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402
from hmm.net_clue import log_ratio  # noqa: E402

STATE = {}
SCALE = 1e6


def _init(state):
    STATE.update(state)
    sf.STATE.update(state)


def configs(variants):
    return sorted({(e, c) for e, c, m in variants.values() if m > 0})


def match_cats(d, ball, cfg):
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    px, py = d["players"][:, :, 0].astype(float), d["players"][:, :, 1].astype(float)
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    keep = ahead.keepers_from_positions(px, py, d["team"], halves, pitch[0])
    usable = ball_is_usable(bx, by, pitch)
    return {(e, c): keeper.keeper_category(bx, by, usable, px, py, keep, pitch, fps, halves,
                                           use=tuple(int(x) for x in c), ends=e)
            for e, c in cfg}


def count_one(name):
    d = sh.load(STATE["paths"][name])
    cats = match_cats(d, match_ball(name, d), STATE["configs"])
    return name, {k: keeper.counts(v, d["labels"]) for k, v in cats.items()}


def run(name):
    paths, tabs, loso, variants = (STATE[k] for k in ("paths", "tables", "loso", "variants"))
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = tabs[k]
    d = sh.load(paths[name])
    ball = match_ball(name, d)
    views = (d["labels"], control_labels(d, STATE["fold_seconds"]))
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
    net = log_ratio(sf.net_probs(name, len(d["halves"]), STATE["tags"]), d["halves"],
                    len(d["labels"]), prior)
    direct = STATE["wc"] * ratio + STATE["wn"] * net
    cats = match_cats(d, ball, STATE["configs"])
    trans, start = dec.transitions(), np.log(dec.START)
    out = {}
    for label, (e, c, m) in variants.items():
        extra = [(width, width_t, dec.WIDTH_WEIGHT)]
        if m > 0:
            extra.append((cats[(e, c)], STATE["keeper"][k][(e, c)], m * dec.WEIGHT))
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), weight=dec.WEIGHT,
                          extra=extra, direct=direct)
        if STATE["hold_restart"]:
            pred = dec.hold_restarts(pred, bx, by, fps)
        pred[d["labels"] == "X"] = "X"
        out[label] = [sh.score(gt, pred) for gt in views]
    return name, out


def parse(spec):
    out = {}
    for item in spec.split(","):
        if item == "0":
            out["v9"] = ("", "", 0.0)
        else:
            e, c, m = item.split(":")
            out[item] = (e, c, float(m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default="0,next:01:4,both:01:4,both:0123:4")
    ap.add_argument("--tag", default="bt,b2")
    ap.add_argument("--wc", type=float, default=0.2)
    ap.add_argument("--wn", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--loso", default="control_loso.npz")
    ap.add_argument("--fold-seconds", type=float, default=sf.CONTROL_SECONDS)
    ap.add_argument("--no-hold-restart", dest="hold_restart", action="store_false")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    variants = parse(a.variants)
    cfg = configs(variants)
    with Pool(a.jobs, initializer=_init, initargs=(dict(paths=paths, configs=cfg),)) as pool:
        cnt = dict(pool.map(count_one, list(paths), chunksize=1))
    tabs = {k: {} for k in sh.SETS}
    for key in cfg:
        per = {}
        for k in sh.SETS:
            h = sum(cnt[n][key][0] for n in cnt if n.startswith(k))
            dd = sum(cnt[n][key][1] for n in cnt if n.startswith(k))
            per[k] = (h / h.sum(), dd / dd.sum())
            print(key, k, "per 1e4 frames  home", np.round(per[k][0][:-1] * 1e4, 1),
                  " dead", np.round(per[k][1][:-1] * 1e4, 1))
        for k in sh.SETS:
            others = [o for o in sh.SETS if o != k]
            t = keeper.table(sum(per[o][0] for o in others) * SCALE,
                             sum(per[o][1] for o in others) * SCALE)
            tabs[k][key] = t
            print(key, "held out", k, "log p(.|H) - log p(.|D):", np.round(t[0] - t[2], 2),
                  " log p(.|A) - log p(.|D):", np.round(t[1] - t[2], 2))
    state = dict(paths=paths, tables=sc.tables(paths), shares=sf.label_shares(paths),
                 loso=dict(np.load(os.path.join(CACHE, a.loso))), variants=variants,
                 tags=a.tag.split(","), fold_seconds=a.fold_seconds, keeper=tabs, configs=cfg,
                 wc=a.wc, wn=a.wn, hold_restart=a.hold_restart)
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        rows = dict(pool.map(run, list(paths), chunksize=1))
    sf.report(rows, next(iter(variants)))
    if a.out:
        labels = list(variants)
        np.savez(os.path.join(CACHE, "keeper_{}.npz".format(a.out)),
                 names=np.array(list(rows)), variants=np.array(labels),
                 scores=np.array([[rows[n][m] for m in labels] for n in rows]))


if __name__ == "__main__":
    main()
