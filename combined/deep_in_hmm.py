"""deep/ as one more clue in hmm v7, held out by provider. An experiment: nothing under hmm/,
deep/ or possession/ changes.

    python external/control_features.py && python external/train_control.py --loso   # once
    python combined/deep_in_hmm.py [--tag v2] [--jobs 1]

hmm v7 is decoded exactly as external/score_control.py does at the shipped weight, with one more
per-frame term added to every state's score: u x (log of deep/'s probability of the state, from
the fold net that never saw the match's provider, minus the log of the class share), the same
form as v7's own control classifier term. u = 0 is hmm v7. For each provider, u is picked by
mean spellF1 on the other two providers (each counting once) and scored on that one. Scored on
the outside labels as they are and folded towards the controlled-possession rule.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import score_control as sc  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402

dec = sc.dec
US = (0.0, 0.05, 0.1, 0.2, 0.4)
# (weight where the ball is usable, weight where it is not); the first is hmm v7
GATED = ((0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.4), (0.0, 0.8), (0.05, 0.2), (0.05, 0.4), (0.05, 0.8))


def one(args):
    name, tag, grid = args
    paths, tabs, loso = (sc.STATE[k] for k in ("paths", "tables", "loso"))
    speed_t, flight_t, gap_t, width_t = tabs[name.split("_")[0]]
    d = sh.load(paths[name])
    ball = sc.match_ball(name, d)
    views = (d["labels"], control_labels(d, sc.STATE["fold_seconds"]))
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t), (sh.flight_cats(d, ball), flight_t)]
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    prior = np.log(loso["prior_" + name])
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - prior
    p = np.load(os.path.join(ROOT, "data", "cache", "deep", tag, name.split("_")[0], name + ".npz"))
    deep = np.zeros_like(ratio)  # nothing outside the halves
    for i, (a, b) in enumerate(d["halves"]):
        deep[a:b + 1] = np.log(np.clip(decode.upsample(p[f"p{i}"], b - a + 1), 1e-6, 1)) - prior
    trans, start = dec.transitions(), np.log(dec.START)
    out = []
    usable = ball_is_usable(bx, by, pitch)
    for u in grid:
        w = np.where(usable, u[0], u[1])[:, None] if isinstance(u, tuple) else u
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), extra=extra,
                          direct=dec.CONTROL_WEIGHT * ratio + w * deep)
        pred[d["labels"] == "X"] = "X"
        out.append([sh.score(gt, pred)[0] for gt in views])
    return name, out, float(usable[d["labels"] != "X"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--gated", action="store_true",
                    help="separate weights where the ball is usable and where it is not")
    a = ap.parse_args()
    grid = GATED if a.gated else US
    fmt = (lambda u: "{:g}/{:g}".format(*u)) if a.gated else "{:g}".format
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(sh.SETS)}
    state = dict(paths=paths, tables=sc.tables(paths), loso=dict(np.load(os.path.join(sc.CACHE, "control_loso.npz"))),
                 weights=[dec.CONTROL_WEIGHT], fold_seconds=sc.CONTROL_SECONDS)
    with Pool(a.jobs, initializer=sc._init, initargs=(state,)) as pool:
        got = list(pool.imap_unordered(one, [(n, a.tag, grid) for n in paths]))
    rows = {n: r for n, r, _ in got}
    print("share of frames in play with a usable ball: " + ", ".join(
        "{} {:.2f}".format(k, np.mean([s for n, _, s in got if n.startswith(k)])) for k in sh.SETS))
    S = {k: np.array([rows[n] for n in rows if n.startswith(k)]) for k in sh.SETS}  # (matches, u, view)
    for v, view in enumerate(("labels as they are", "folded labels")):
        print(f"\n{view}: spellF1 by u = " + ", ".join(fmt(u) for u in grid))
        for k in sh.SETS:
            print(f"  {k:12s} " + "  ".join(f"{x:.3f}" for x in S[k][:, :, v].mean(0)))
        print(f"  {'provider mean':12s} " + "  ".join(f"{x:.3f}" for x in np.mean([S[k][:, :, v].mean(0) for k in sh.SETS], 0)))
        print("  held out (u picked on the other two providers):")
        got = []
        for k in sh.SETS:
            pick = int(np.argmax(np.mean([S[o][:, :, v].mean(0) for o in sh.SETS if o != k], 0)))
            diff = S[k][:, pick, v] - S[k][:, 0, v]
            got.append((S[k][:, pick, v].mean(), S[k][:, 0, v].mean()))
            print(f"    {k:12s} u={fmt(grid[pick]):<8s} {got[-1][0]:.3f} against hmm v7 {got[-1][1]:.3f}  "
                  f"({diff.mean():+.3f}, se {diff.std(ddof=1) / np.sqrt(len(diff)):.3f}, up {int((diff > 0).sum())}/{len(diff)})")
        g = np.mean(got, 0)
        print(f"    {'mean':12s}         {g[0]:.3f} against hmm v7 {g[1]:.3f}")


if __name__ == "__main__":
    main()
