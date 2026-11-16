"""hmm_v9 plus the look-ahead ball clue (hmm/ahead.py), scored as external/score_fuse.py scores v9.

    python db/score_ahead.py --tag bt,b2 [--mults 0,0.5,1,2]

Each provider is decoded with an ahead table counted on the other two providers' labels (the
ball as the scorer sees it), so no match is read by a table that saw its provider. Variant 0 is
v9 exactly (control weight 0.2, net weight 0.3).
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
from hmm import ahead  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402
from hmm.net_clue import log_ratio  # noqa: E402

STATE = {}
HIDE = {"on": False}


def keeper_hidden(name, d):
    """The scorer's ball, also hidden wherever it is within 1 m of a keeper inside his box (and
    for 1 s either side), the way BAR-ATH's tracker loses it in the keeper's hands. A scenario,
    not the yardstick."""
    ball = match_ball_scorer(name, d)
    if not HIDE["on"]:
        return ball
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    px, py = d["players"][:, :, 0].astype(float), d["players"][:, :, 1].astype(float)
    keep = ahead.keepers_from_positions(px, py, d["team"], halves, pitch[0])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    hold = np.zeros(len(bx), bool)
    f = np.arange(len(bx))
    for side in (0, 1):
        k = keep[:, side]
        ok = k >= 0
        kx = np.where(ok, px[f, np.maximum(k, 0)], np.nan)
        ky = np.where(ok, py[f, np.maximum(k, 0)], np.nan)
        with np.errstate(invalid="ignore"):
            hold |= ((np.hypot(kx - bx, ky - by) <= 1.0)
                     & (np.minimum(kx, pitch[0] - kx) <= ahead.BOX_DEPTH)
                     & (np.abs(ky - pitch[1] / 2) <= ahead.BOX_HALF_WIDTH))
    hold = np.convolve(hold, np.ones(2 * fps + 1), "same") > 0
    out = ball.copy()
    out[hold] = np.nan
    return out


match_ball_scorer = match_ball
match_ball = keeper_hidden  # noqa: F811


def _init(state):
    STATE.update(state)
    HIDE["on"] = state.get("hide", False)
    sf.STATE.update(state)


def match_cats(name, d, ball):
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    px, py = d["players"][:, :, 0].astype(float), d["players"][:, :, 1].astype(float)
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    keep = ahead.keepers_from_positions(px, py, d["team"], halves, pitch[0])
    return ahead.ahead_category(bx, by, ball_is_usable(bx, by, pitch), px, py, keep, pitch, fps,
                                halves)


def count_one(item):
    name, path = item
    d = sh.load(path)
    cats = match_cats(name, d, match_ball(name, d))
    return name, ahead.counts(cats, d["labels"])


def run(name):
    paths, tabs, loso, mults = (STATE[k] for k in ("paths", "tables", "loso", "mults"))
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
    cats = match_cats(name, d, ball)
    trans, start = dec.transitions(), np.log(dec.START)
    out = {}
    for m in mults:
        extra = [(width, width_t, dec.WIDTH_WEIGHT)]
        if m > 0:
            extra.append((cats, STATE["ahead"][k], m * dec.WEIGHT))
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), weight=dec.WEIGHT,
                          extra=extra, direct=direct)
        pred[d["labels"] == "X"] = "X"
        out["ahead x%g" % m] = [sh.score(gt, pred) for gt in views]
        if STATE["save"] and m == mults[-1]:
            np.save(os.path.join(CACHE, "ahead_pred_%s_%g.npy" % (name, m)), pred)
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mults", default="0,0.5,1,2,3")
    ap.add_argument("--tag", default="bt,b2")
    ap.add_argument("--wc", type=float, default=0.2)
    ap.add_argument("--wn", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--loso", default="control_loso.npz")
    ap.add_argument("--fold-seconds", type=float, default=sf.CONTROL_SECONDS)
    ap.add_argument("--out", default="")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--keeper-hide", action="store_true", help="scenario: hide the ball in keepers' hands")
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    HIDE["on"] = a.keeper_hide
    with Pool(a.jobs, initializer=_init, initargs=(dict(hide=a.keeper_hide),)) as pool:
        cnt = dict(pool.map(count_one, list(paths.items()), chunksize=1))
    per = {}
    for k in sh.SETS:
        h = sum(cnt[n][0] for n in cnt if n.startswith(k))
        dd = sum(cnt[n][1] for n in cnt if n.startswith(k))
        per[k] = (h / max(h.sum(), 1), dd / max(dd.sum(), 1))
        print(k, "home", np.round(per[k][0], 4), "\n   dead", np.round(per[k][1], 4))
    tabs_ahead = {}
    for k in sh.SETS:
        others = [o for o in sh.SETS if o != k]
        h = sum(per[o][0] for o in others) * 1e3
        dd = sum(per[o][1] for o in others) * 1e3
        tabs_ahead[k] = ahead.table(h, dd)
        print(k, "log p(next|last,H) - log p(next|last,D):",
              np.round(tabs_ahead[k][0] - tabs_ahead[k][2], 2))
    mults = [float(x) for x in a.mults.split(",")]
    state = dict(paths=paths, tables=sc.tables(paths), shares=sf.label_shares(paths),
                 loso=dict(np.load(os.path.join(CACHE, a.loso))), mults=mults,
                 tags=a.tag.split(","), fold_seconds=a.fold_seconds, ahead=tabs_ahead,
                 wc=a.wc, wn=a.wn, save=a.save, hide=a.keeper_hide)
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        rows = dict(pool.map(run, list(paths), chunksize=1))
    sf.report(rows, "ahead x%g" % mults[0])
    if a.out:
        labels = ["ahead x%g" % m for m in mults]
        np.savez(os.path.join(CACHE, "ahead_{}.npz".format(a.out)),
                 names=np.array(list(rows)), variants=np.array(labels),
                 scores=np.array([[rows[n][m] for m in labels] for n in rows]))


if __name__ == "__main__":
    main()
