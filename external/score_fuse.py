"""Score hmm_v8 with the players-only net of deep/ added to its emissions (the v9 candidate).

    python external/control_features.py && python external/train_control.py --loso
    sh deep/queue.sh deep/jobs.txt                      # the bt fold nets (first three lines)
    python external/score_fuse.py [--variants "0.2:0,0.2:0.1,..."] [--tag bt]

Each variant is "control weight:net weight[:width weight[:clue weight[:net weight where the
ball is unseen[:multiple on the net's alive/dead part]]]]"; -1 keeps a default. The net's per-frame
p(H, A, D) for a match comes from the fold net that never saw the match's provider
(data/cache/deep/<tag>/<provider>/), turned into a log ratio against the class shares of the
providers it was trained on and added to the decoder's log emissions next to the control
classifier's. Everything else is external/score_control.py: same ball, same tables, same folded
yardstick. Prints aggregates only, and saves every match's scores to data/cache/fuse_<out>.npz.
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
import score_hmm as sh  # noqa: E402
from control import CONTROL_SECONDS, control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.net_clue import log_ratio  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402

STATE = {}


def _init(state):
    STATE.update(state)


def net_probs(name, n_halves, tags):
    """Per tag, the held-out fold net's (T, 3) p(H, A, D) for each half of a match."""
    out = []
    for tag in tags:
        z = np.load(os.path.join(CACHE, "deep", tag, name.split("_")[0], name + ".npz"))
        out.append([z["p%d" % i] for i in range(n_halves)])
    return out


def label_shares(paths):
    """Per provider, the H / A / D shares of its graded frames (H and A made equal)."""
    out = {}
    for k in sh.SETS:
        rows = []
        for n, p in paths.items():
            if n.startswith(k):
                lab = np.load(p)["labels"]
                c = np.array([np.sum(lab == s) for s in "HAD"], float)
                rows.append(c / c.sum())
        s = np.mean(rows, axis=0)
        out[k] = np.array([(s[0] + s[1]) / 2, (s[0] + s[1]) / 2, s[2]])
    return out


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
    net = log_ratio(net_probs(name, len(d["halves"]), STATE["tags"]), d["halves"],
                    len(d["labels"]), prior)
    trans, start = dec.transitions(), np.log(dec.START)
    fold = int(round(STATE["min_possession"] * fps))
    out = {}
    seen = ball_is_usable(bx, by, pitch)[:, None]
    # the net's log ratio split into its alive/dead part (the same for H and A) and its team part
    live = np.logaddexp(net[:, 0], net[:, 1]) - np.log(2)
    net_dead = np.column_stack([live, live, net[:, 2]])
    net_team = net - net_dead
    for label, v in variants.items():
        wc, wn, ww, wk = v[:4]
        wu = v[4] if len(v) > 4 else wn      # net weight where the ball is not seen
        wd = v[5] if len(v) > 5 else 1.0     # the alive/dead part's weight, as a multiple
        direct = wc * ratio + np.where(seen, wn, wu) * (net_team + wd * net_dead)
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), weight=wk,
                          extra=[(width, width_t, ww)], direct=direct, min_possession=fold)
        pred[d["labels"] == "X"] = "X"
        out[label] = [sh.score(gt, pred) for gt in views]
    return name, out


def parse(spec):
    out = {}
    for item in spec.split(","):
        v = [float(x) for x in item.split(":")]
        v += [dec.WIDTH_WEIGHT, dec.WEIGHT][len(v) - 2:] if len(v) < 4 else []
        v = [dec.WIDTH_WEIGHT if x < 0 else x for x in v[:3]] + [dec.WEIGHT if x < 0 else x for x in v[3:4]] + v[4:]
        out[item] = tuple(v)
    return out


def report(rows, base):
    models = list(next(iter(rows.values())))
    means = {m: [] for m in models}
    for k in (k for k in sh.SETS if any(n.startswith(k) for n in rows)):
        names = [n for n in rows if n.startswith(k)]
        print("\n{} ({} matches): spellF1 raw | folded (change from {} paired by match, se, up) | "
              "alive/dead, team acc (folded) | possessions".format(k, len(names), base))
        for m in models:
            raw = np.mean([rows[n][m][0] for n in names], axis=0)
            fol = np.mean([rows[n][m][1] for n in names], axis=0)
            diff = np.array([rows[n][m][1][0] - rows[n][base][1][0] for n in names])
            means[m].append((raw[0], fol[0], fol[2], fol[3]))
            print("  {:22s} {:.3f} | {:.3f} ({:+.3f}, se {:.3f}, up {}/{}) | {:.3f} {:.3f} | {:.0f}"
                  .format(m, raw[0], fol[0], diff.mean(),
                          diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0,
                          int(np.sum(diff > 0)), len(diff), fol[2], fol[3], raw[6]))
    print("\nmean of the providers: raw | folded | alive/dead | team acc")
    for m in models:
        print("  {:22s} {:.3f} | {:.4f} | {:.3f} | {:.3f}   ({})".format(
            m, *np.mean(means[m], axis=0), " / ".join("%.3f" % x[1] for x in means[m])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default="0.2:0,0:0.2,0.2:0.05,0.2:0.1,0.2:0.2,0.1:0.2,0.2:0.3")
    ap.add_argument("--tag", default="bt", help="comma-separated deep/ tags to average")
    ap.add_argument("--sets", default=",".join(sh.SETS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--loso", default="control_loso.npz")
    ap.add_argument("--min-possession", type=float, default=dec.MIN_POSSESSION_SECONDS)
    ap.add_argument("--fold-seconds", type=float, default=CONTROL_SECONDS)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(tuple(a.sets.split(",")))}
    all_paths = {os.path.basename(p)[:-4]: p
                 for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    variants = parse(a.variants)
    state = dict(paths=paths, tables=sc.tables(all_paths), shares=label_shares(all_paths),
                 loso=dict(np.load(os.path.join(CACHE, a.loso))), variants=variants,
                 tags=a.tag.split(","), fold_seconds=a.fold_seconds,
                 min_possession=a.min_possession)
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        rows = dict(pool.map(run, list(paths), chunksize=1))
    report(rows, next(iter(variants)))
    if a.out:
        np.savez(os.path.join(CACHE, "fuse_{}.npz".format(a.out)),
                 names=np.array(list(rows)), variants=np.array(list(variants)),
                 scores=np.array([[rows[n][m] for m in variants] for n in rows]))


if __name__ == "__main__":
    main()
