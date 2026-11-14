"""Score hmm_v8 (v6 plus the control classifier) against hmm_v6 on the external matches.

    python external/control_features.py
    python external/train_control.py --loso         # needs scikit-learn
    python external/score_control.py [--weights 0.05,0.1,0.2] [--sets ...]

The classifier's posteriors come from external/train_control.py --loso: each set is scored with a
model trained on the other two, so no match is scored by a classifier that saw it. Everything else
is as in external/score_hmm.py (same ball, same tables, each provider's tables counted on the
other). Each variant is scored twice: against the labels as they are, and against the labels
folded towards the challenge's controlled-possession rule (external/control.py), since the
external labels count touch-level spells the challenge does not. Prints aggregates only.
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
import score_hmm as sh  # noqa: E402
from control import CONTROL_SECONDS, control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402

STATE = {}


def _init(state):
    """Pool initializer: macOS starts workers fresh (spawn), so they get the state here."""
    STATE.update(state)


def tables(paths):
    """Per set: (speed, flight, gap-distance, width) tables, as external/score_hmm.py counts them."""
    matches = {n: sh.load(p) for n, p in paths.items() if n.startswith(sh.PROVIDERS)}
    balls = {n: match_ball(n, d) for n, d in matches.items()}
    by = {k: [d for n, d in matches.items() if n.startswith(k)] for k in sh.PROVIDERS}
    out = {}
    for k, o in zip(sh.PROVIDERS, sh.PROVIDERS[::-1]):
        names = [n for n in matches if n.startswith(o)]
        ms, bs = [matches[n] for n in names], [balls[n] for n in names]
        out[k] = [dec.speed_table(*sh.speed_given_state(by[o])),
                  dec.flight_table(sh.flight_given_state(ms, bs)),
                  dec.gap_distance_table(*sh.gap_distance_given_state(ms, bs))]
    out[sh.HELD_OUT] = [dec.speed_table(), dec.flight_table(), dec.gap_distance_table()]
    shares = {n: sh.width_given_state(sh.load(p)) for n, p in paths.items()}
    for k in sh.SETS:
        out[k].append(dec.width_table(*sh.width_over_sets(shares, [o for o in sh.SETS if o != k])))
    return out


def run(name):
    paths, tabs, loso, weights = (STATE[k] for k in ("paths", "tables", "loso", "weights"))
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
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - np.log(loso["prior_" + name])
    trans, start = dec.transitions(), np.log(dec.START)
    out = {}
    fold = int(round(STATE["min_possession"] * fps))
    for w in (0.0,) + tuple(weights):
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), extra=extra,
                          direct=w * ratio if w else None, min_possession=fold if w else 0)
        pred[d["labels"] == "X"] = "X"
        out["hmm_v6" if w == 0 else "v8 w{:g}".format(w)] = [sh.score(gt, pred) for gt in views]
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="0.1,0.15,0.2,0.3")
    ap.add_argument("--sets", default=",".join(sh.SETS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--loso", default="control_loso.npz",
                    help="posteriors file in the cache, from train_control.py --loso [--out]")
    ap.add_argument("--min-possession", type=float, default=dec.MIN_POSSESSION_SECONDS,
                    help="seconds under which the decoder folds a possession between two spells "
                         "of the other team (default {}: none)".format(dec.MIN_POSSESSION_SECONDS))
    ap.add_argument("--fold-seconds", type=float, default=CONTROL_SECONDS,
                    help="spell length under which the folded view folds (external/control.py)")
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(tuple(a.sets.split(",")))}
    loso_path = os.path.join(CACHE, a.loso)
    if not os.path.exists(loso_path):
        sys.exit("no {}: run external/train_control.py --loso first".format(loso_path))
    state = dict(paths=paths, tables=tables(paths), loso=dict(np.load(loso_path)),
                 weights=[float(w) for w in a.weights.split(",")], fold_seconds=a.fold_seconds,
                 min_possession=a.min_possession)
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        rows = dict(pool.map(run, list(paths)))
    models = list(next(iter(rows.values())))
    for k in (k for k in sh.SETS if any(n.startswith(k) for n in rows)):
        names = [n for n in rows if n.startswith(k)]
        print("\n{} ({} matches): spellF1, its change from hmm_v6 paired by match (standard error, "
              "matches up), predicted possessions".format(k, len(names)))
        for m in models:
            line = "  {:9s}".format(m)
            for v, tag in ((0, "raw"), (1, "folded")):
                mean = np.mean([rows[n][m][v] for n in names], axis=0)
                diff = np.array([rows[n][m][v][0] - rows[n]["hmm_v6"][v][0] for n in names])
                line += "  | {} {:.3f} ({:+.3f}, se {:.3f}, up {}/{})".format(
                    tag, mean[0], diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)),
                    int(np.sum(diff > 0)), len(diff))
            print(line + "  | possessions {:.0f}".format(
                np.mean([rows[n][m][0][6] for n in names])))


if __name__ == "__main__":
    main()
