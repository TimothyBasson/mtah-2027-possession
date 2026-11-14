"""Score label-free, per-match corrections of the control classifier's clue in the decoder.

    python external/control_features.py
    python external/train_control.py --loso
    python external/train_calib.py
    python external/score_calib.py [--weight 0.15] [--scales 0.5,1,2]

hmm_v7 adds  w * (log p(state | x) - log training share)  to the log emissions. That is a
likelihood ratio, so the target match's own class shares cancel out of it: re-estimating them (EM,
Saerens et al. 2002) and renormalising changes nothing for Viterbi. A correction therefore only
acts as a per-match offset per state, or as a rescaling of log p. The variants, all without the
target match's labels:

  T         log p rescaled by 1/T, T fitted on the two training providers, each scored by a
            classifier trained on the other alone (external/train_calib.py)
  em        offset  log(EM share) - log(training share)
  emD       the same offset for dead against alive only (home and away kept level)
  Tem, TemD em and emD run on the tempered posteriors
  src       offset  log(training share) - log(mean posterior): pull the classifier's average
            call on this match back to the training shares
  v6        offset  log(hmm_v6 path share) - log(mean posterior): pull it to what the decoder
            says without the classifier
  v7        the same with hmm_v7's own first-pass path (one round of self-training)
  oracle    NOT label-free, a ceiling only: offset  log(true share) - log(mean posterior) on the
            graded frames. What a perfect estimate of the match's shares would be worth. Left out
            of the choice in the last block.

Each offset is tried at the --scales, inside the weight: w * (ratio + scale * offset). The shares
are taken over the frames inside the halves, graded or not. Scores are against the folded labels
(external/control.py), as the v7 numbers are. The last block picks, for each provider, the variant
that does best on the *other two* and reports it on that provider: the honest number, since the
choice never sees the provider it is scored on.
"""

import argparse
import glob
import json
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
from control import control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402

STATE = {}
FLOOR = 1e-6


def _init(state):
    STATE.update(state)


def temper(p, t):
    z = np.log(np.clip(p, FLOOR, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p, y):
    """The T that minimises the log loss of temper(p, T) on labels y (a 1-d grid search)."""
    grid = np.exp(np.linspace(np.log(0.5), np.log(8), 57))
    loss = [-np.mean(np.log(np.clip(temper(p, t)[np.arange(len(y)), y], 1e-12, 1))) for t in grid]
    return float(grid[int(np.argmin(loss))])


def temperatures(inner, sets):
    """Per left-out set: T fitted on the other two, each scored by the other's classifier."""
    out = {}
    for k in sets:
        ps, ys = [], []
        for key in inner:
            if key.startswith("y__"):
                continue
            a, n = key.split("__")
            if a == k or n.startswith(k):
                continue
            ps.append(inner[key])
            ys.append(inner["y__" + n])
        out[k] = fit_temperature(np.vstack(ps), np.concatenate(ys).astype(int))
    return out


def em_shares(p, prior, tie=False, n_iter=200):
    """Saerens' EM: the class shares of this match that the posteriors p (from a classifier with
    training shares `prior`) agree with. tie keeps home and away at their training ratio."""
    pi = prior.copy()
    for _ in range(n_iter):
        q = p * (pi / prior)
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(axis=0)
        if tie:
            alive = new[0] + new[1]
            new = np.array([alive * prior[0] / (prior[0] + prior[1]),
                            alive * prior[1] / (prior[0] + prior[1]), new[2]])
        if np.abs(new - pi).max() < 1e-7:
            return new
        pi = new
    return pi


def path_shares(pred, inside):
    s = np.array([np.mean(pred[inside] == c) for c in "HAD"])
    return (s + 0.01) / (s + 0.01).sum()


def run(name):
    paths, tabs, loso = (STATE[k] for k in ("paths", "tables", "loso"))
    w, scales, temps = STATE["weight"], STATE["scales"], STATE["temps"]
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = tabs[k]
    d = sh.load(paths[name])
    ball = match_ball(name, d)
    gt = control_labels(d)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    inside = np.zeros(len(gt), bool)
    for lo, hi in segments:
        inside[lo:hi] = True
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t),
             (sh.flight_cats(d, ball), flight_t)]
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    trans, start = dec.transitions(), np.log(dec.START)

    def decode(direct):
        pred = dec.decode(clues, trans, start, segments, kickoffs=(), extra=extra, direct=direct)
        pred[d["labels"] == "X"] = "X"
        return pred

    p = np.clip(loso[name].astype(float), FLOOR, 1)
    prior = loso["prior_" + name].astype(float)
    log_prior = np.log(prior)
    ratio = np.log(p) - log_prior
    p_t = np.clip(temper(p, temps[k]), FLOOR, 1)
    ratio_t = np.log(p_t) - log_prior
    graded = gt != "X"
    true = np.array([np.mean(gt[graded] == c) for c in "HAD"])
    v6 = decode(None)
    v7 = decode(w * ratio)
    mean_p, mean_pt = p[inside].mean(axis=0), p_t[inside].mean(axis=0)
    em = {"em": em_shares(p[inside], prior), "emD": em_shares(p[inside], prior, tie=True),
          "Tem": em_shares(p_t[inside], prior), "TemD": em_shares(p_t[inside], prior, tie=True)}
    offsets = {
        "em": (ratio, np.log(em["em"]) - log_prior),
        "emD": (ratio, np.log(em["emD"]) - log_prior),
        "Tem": (ratio_t, np.log(em["Tem"]) - log_prior),
        "TemD": (ratio_t, np.log(em["TemD"]) - log_prior),
        "src": (ratio, log_prior - np.log(mean_p)),
        "v6": (ratio, np.log(path_shares(v6, inside)) - np.log(mean_p)),
        "v7": (ratio, np.log(path_shares(v7, inside)) - np.log(mean_p)),
        "oracle": (ratio, np.log(true) - np.log(p[graded].mean(axis=0))),
    }
    preds = {"hmm_v6": v6, "v7": v7, "T": decode(w * ratio_t)}
    for tag, (base, off) in offsets.items():
        for s in scales:
            preds["{} x{:g}".format(tag, s)] = decode(w * (base + s * off))
    info = dict(true=true.tolist(),
                mean_p=mean_p.tolist(), mean_p_graded=p[graded].mean(axis=0).tolist(),
                em=em["em"].tolist(), Tem=em["Tem"].tolist(),
                v6=path_shares(v6, inside).tolist(), prior=prior.tolist(),
                logloss=float(-np.mean(np.log(p[graded, sh_code(gt[graded])]))),
                logloss_T=float(-np.mean(np.log(p_t[graded, sh_code(gt[graded])]))))
    return name, {m: [float(v) for v in sh.score(gt, pred)] for m, pred in preds.items()}, info


def sh_code(labels):
    return np.select([labels == "H", labels == "A"], [0, 1], 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weight", type=float, default=dec.CONTROL_WEIGHT)
    ap.add_argument("--scales", default="0.5,1,2")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--loso", default="control_loso.npz")
    ap.add_argument("--out", default="calib_scores.json", help="per-match scores, in the cache")
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(sh.SETS)}
    temps = temperatures(dict(np.load(os.path.join(CACHE, "control_inner.npz"))), sh.SETS)
    print("temperature per left-out set:", {k: round(v, 2) for k, v in temps.items()})
    state = dict(paths=paths, tables=sc.tables(paths), weight=a.weight, temps=temps,
                 loso=dict(np.load(os.path.join(CACHE, a.loso))),
                 scales=[float(s) for s in a.scales.split(",")])
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        res = pool.map(run, list(paths))
    rows = {n: r for n, r, _ in res}
    info = {n: i for n, _, i in res}
    with open(os.path.join(CACHE, a.out), "w") as handle:
        json.dump(dict(rows=rows, info=info, temps=temps), handle)
    models = list(next(iter(rows.values())))
    by = {k: [n for n in rows if n.startswith(k)] for k in sh.SETS}

    print("\nshares H/A/D per set (mean over matches) and held-out log loss on graded frames")
    for k in sh.SETS:
        for key in ("true", "prior", "mean_p", "mean_p_graded", "em", "Tem", "v6"):
            v = np.mean([info[n][key] for n in by[k]], axis=0)
            print("  {:12s} {:14s} {:.3f} {:.3f} {:.3f}".format(k, key, *v))
        print("  {:12s} log loss {:.3f}, tempered {:.3f}".format(
            k, np.mean([info[n]["logloss"] for n in by[k]]),
            np.mean([info[n]["logloss_T"] for n in by[k]])))

    mean = {m: {k: np.mean([rows[n][m][0] for n in by[k]]) for k in sh.SETS} for m in models}
    print("\nfolded spellF1 per set, mean, change from v7 paired by match (se) per set, pred dead")
    for m in models:
        line = "  {:10s}".format(m) + " ".join("{:.3f}".format(mean[m][k]) for k in sh.SETS)
        line += "  mean {:.3f} |".format(np.mean(list(mean[m].values())))
        for k in sh.SETS:
            diff = np.array([rows[n][m][0] - rows[n]["v7"][0] for n in by[k]])
            line += " {:+.3f} ({:.3f}, up {}/{})".format(
                diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)), int(np.sum(diff > 0)),
                len(diff))
        line += " | D " + " ".join("{:.2f}".format(np.mean([rows[n][m][5] for n in by[k]]))
                                   for k in sh.SETS)
        print(line)

    print("\nvariant chosen on the other two sets, scored on the set it did not see")
    picked = []
    for k in sh.SETS:
        others = [o for o in sh.SETS if o != k]
        best = max((m for m in models if not m.startswith("oracle")), key=lambda m: np.mean([mean[m][o] for o in others]))
        picked.append(mean[best][k])
        print("  {:12s} {:10s} {:.3f} (v7 {:.3f})".format(k, best, mean[best][k], mean["v7"][k]))
    print("  mean {:.3f} (v7 {:.3f})".format(
        np.mean(picked), np.mean([mean["v7"][k] for k in sh.SETS])))


if __name__ == "__main__":
    main()
