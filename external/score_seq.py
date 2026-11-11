"""Score the learned labeller (seq/) on the out-of-sample matches, leaving each set out in turn.

    python external/score_seq.py [--seen 0.59] [--drop k_] [--tag name] [--refit]

Each set (IDSSE, Metrica, SkillCorner) is scored by a model fitted on the other two only, the two
counting equally. The chain's numbers are chosen the same way: for the set being scored, the
setting that does best on the other two sets' left-out predictions, so nothing about a set's own
score picks its setting. The ball is the one external/score_hmm.py gives its models (same seed),
so the rows compare with that table.

--drop   leave out feature groups by prefix (p_ players, b_ ball against players, k_ ball alone).
         The default leaves out k_: see seq/model.py for why.
--tag    name for the cached probabilities under data/cache/seq/, so variants can sit side by side

Two models are scored. "seq" is the learned labeller on its own chain. "hybrid" is hmm_v6 with the
labeller's per-frame log-probabilities added to its clues at a weight, decoded by hmm_v6's own
chain at 25 fps; at weight 0 it is hmm_v6 itself, which is the row to compare against. The HMM's
tables are counted as in external/score_hmm.py (each provider with the other's, SkillCorner with
the shipped constants), and the weight is chosen for each set on the other two.
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_metrics as pm  # noqa: E402
from score import drop_ball  # noqa: E402
import hmm.decode as dec  # noqa: E402
import score_hmm as sh  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402
from hmm.model import STATES, build_log_emit, viterbi  # noqa: E402
from seq import model  # noqa: E402

SETS = ("idsse", "metrica", "skillcorner")
CACHE = os.path.join(ROOT, "data", "cache", "seq")
GRID = dict(weight=(0.15, 0.25, 0.4, 0.6, 1.0), possession=(4.0, 8.0, 12.0, 20.0),
            stoppage=(4.0, 8.0, 16.0))
HYBRID_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.4)


def hmm_emissions(paths, seen):
    """name -> path of hmm_v6's per-frame log-emissions (n, 3), cached. The tables are counted as
    in score_hmm.py's main()."""
    out = os.path.join(CACHE, "hmm_emit_{:.2f}".format(seen))
    os.makedirs(out, exist_ok=True)
    files = {n: os.path.join(out, n + ".npy") for n in paths}
    if all(os.path.exists(f) for f in files.values()):
        return files
    matches = {n: sh.load(p) for n, p in paths.items() if n.startswith(sh.PROVIDERS)}
    width_shares = {n: sh.width_given_state(sh.load(p)) for n, p in paths.items()}
    balls = {n: ball_of(n, d, seen) for n, d in matches.items()}
    speed, flight, gap = {}, {}, {}
    for k, o in zip(sh.PROVIDERS, sh.PROVIDERS[::-1]):
        names = [n for n in matches if n.startswith(o)]
        ds, bs = [matches[n] for n in names], [balls[n] for n in names]
        speed[k] = dec.speed_table(*sh.speed_given_state(ds))
        flight[k] = dec.flight_table(sh.flight_given_state(ds, bs))
        gap[k] = dec.gap_distance_table(*sh.gap_distance_given_state(ds, bs))
    speed[sh.HELD_OUT], flight[sh.HELD_OUT] = dec.speed_table(), dec.flight_table()
    gap[sh.HELD_OUT] = dec.gap_distance_table()
    width = {k: dec.width_table(*sh.width_over_sets(width_shares, [o for o in SETS if o != k]))
             for k in SETS}
    for n, p in paths.items():
        d = matches.get(n) or sh.load(p)
        ball = balls[n] if n in balls else ball_of(n, d, seen)
        k, fps = n.split("_")[0], int(d["fps"])
        pitch = tuple(float(v) for v in d["pitch"])
        bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
        d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
        usable = ball_is_usable(bx, by, pitch)
        place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
        clues = [(gap_distance_category(usable, d_home, d_away), gap[k], dec.WEIGHT),
                 (sh.speed_cats(d), speed[k], dec.WEIGHT),
                 (sh.flight_cats(d, ball), flight[k], dec.WEIGHT),
                 (place, dec.place_table(), dec.WEIGHT),
                 (sh.width_cats(d), width[k], dec.WIDTH_WEIGHT)]
        np.save(files[n], build_log_emit(clues).astype(np.float32))
    return files


def hybrid_labels(hmm_emit, prob, frames, halves, weight, stride=model.STRIDE):
    """hmm_v6's chain over its own clues plus weight * the labeller's log-probabilities."""
    n = len(hmm_emit)
    log_prob = np.log(np.clip(prob, 1e-6, 1.0))[
        np.clip((np.arange(n) + stride // 2) // stride, 0, len(frames) - 1)]
    trans, start = dec.transitions(), np.log(dec.START)
    labels = np.full(n, "X", dtype="<U1")
    for lo, hi in halves:
        labels[lo:hi + 1] = STATES[viterbi(hmm_emit[lo:hi + 1] + weight * log_prob[lo:hi + 1],
                                           trans, start)]
    return labels


def ball_of(name, d, seen):
    if name.startswith("idsse"):
        return drop_ball(d["ball"], seen, int(d["fps"]))
    if name.startswith("skillcorner"):
        return drop_ball(d["ball"], seen, int(d["fps"]), labels=d["labels"])
    return d["ball"]


def features_of(name, path, seen):
    """(X, X_swapped, names, frames, labels at those frames), cached."""
    out = os.path.join(CACHE, "feat_{}_{:.2f}.npz".format(name, seen))
    if os.path.exists(out):
        z = np.load(out)
        return z["X"], z["Xs"], list(z["names"]), z["frames"], z["y"]
    d = dict(np.load(path))
    ball = ball_of(name, d, seen)
    X, names, frames = model.featurise(d, ball)
    Xs = model.featurise(d, ball, swap=True)[0]
    y = d["labels"][frames]
    np.savez(out, X=X, Xs=Xs, names=np.array(names), frames=frames, y=y)
    return X, Xs, names, frames, y


def spell_f1(gt, pred):
    pred = pred.copy()
    pred[gt == "X"] = "X"
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"],
                     r["team_acc_both_alive"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seen", type=float, default=0.59)
    ap.add_argument("--drop", default="k_")
    ap.add_argument("--tag", default="nok")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--max-iter", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(SETS)}
    prob_path = os.path.join(CACHE, "prob_{}_{:.2f}.npz".format(a.tag, a.seen))
    meta = {}
    for name, path in paths.items():
        d = np.load(path)
        meta[name] = dict(labels=d["labels"], halves=[tuple(h) for h in d["halves"]],
                          fps=int(d["fps"]))

    if a.refit or not os.path.exists(prob_path):
        drop = tuple(p for p in a.drop.split(",") if p)
        feats = {}
        for name, path in paths.items():
            feats[name] = features_of(name, path, a.seen)
            print("features", name, feats[name][0].shape, flush=True)
        names = feats[next(iter(feats))][2]
        keep = np.array([not n.startswith(drop) if drop else True for n in names])
        print("{} of {} features kept".format(keep.sum(), len(keep)))
        probs = {}
        for held in SETS:
            Xs, ys, ws = [], [], []
            for other in SETS:
                if other == held:
                    continue
                rows = sum(np.sum(feats[n][4][::2] != "X") for n in feats if n.startswith(other))
                for n in feats:
                    if not n.startswith(other):
                        continue
                    X, Xsw, _, _, y = feats[n]
                    ok = np.flatnonzero(y != "X")[::2]          # 2.5 Hz, both ways round
                    code = np.array([model.CODES[c] for c in y[ok]])
                    Xs += [X[ok][:, keep], Xsw[ok][:, keep]]
                    ys += [code, np.where(code == 2, 2, 1 - code)]
                    ws += [np.full(2 * len(ok), 1.0 / rows)]
            Xtr, ytr, wtr = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
            print("fit without {}: {} rows".format(held, len(ytr)), flush=True)
            clf = model.fit(Xtr, ytr, wtr * len(wtr) / wtr.sum(), max_iter=a.max_iter)
            del Xtr
            for n in feats:
                if n.startswith(held):
                    probs[n] = model.probabilities(clf, feats[n][0][:, keep], feats[n][1][:, keep])
        np.savez(prob_path, **probs, **{"frames_" + n: feats[n][3] for n in feats})
    z = np.load(prob_path)
    probs = {n: z[n] for n in paths}
    frames = {n: z["frames_" + n] for n in paths}

    # per-frame quality before any decoding
    for k in SETS:
        acc = []
        for n in paths:
            if n.startswith(k):
                y = meta[n]["labels"][frames[n]]
                ok = y != "X"
                acc.append(np.mean(model.STATES[np.argmax(probs[n][ok], axis=1)] == y[ok]))
        print("{:11s} frame accuracy of the scorer alone {:.3f}".format(k, np.mean(acc)))

    def run(name, **kw):
        m = meta[name]
        pred = model.decode(probs[name], frames[name], len(m["labels"]), m["halves"], m["fps"],
                            **kw)
        return spell_f1(m["labels"], pred)

    keys = list(GRID)
    settings = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    table = {n: np.array([run(n, **s)[0] for s in settings]) for n in paths}
    by_set = {k: np.mean([table[n] for n in paths if n.startswith(k)], axis=0) for k in SETS}
    print("\nseq alone: left-out scores, chain chosen on the other two sets:")
    for k in SETS:
        others = np.mean([by_set[o] for o in SETS if o != k], axis=0)
        best = int(np.argmax(others))
        rows = np.array([run(n, **settings[best]) for n in paths if n.startswith(k)])
        print("  {:11s} spellF1 {:.3f}  macroF1 {:.3f}  alive/dead {:.3f}  team_acc {:.3f}   "
              "{}   (own best {:.3f}, worst match {:.3f})".format(
                  k, *rows.mean(axis=0), settings[best], by_set[k].max(), rows[:, 0].min()))
    every = np.mean([by_set[k] for k in SETS], axis=0)
    print("  one setting for all three: {} -> {}".format(
        settings[int(np.argmax(every))],
        "  ".join("{} {:.3f}".format(k, by_set[k][int(np.argmax(every))]) for k in SETS)))

    emit = hmm_emissions(paths, a.seen)
    hyb = {n: np.array([spell_f1(meta[n]["labels"], hybrid_labels(
        np.load(emit[n]).astype(float), probs[n], frames[n], meta[n]["halves"], w))
        for w in HYBRID_WEIGHTS]) for n in paths}
    by_set = {k: np.mean([hyb[n] for n in paths if n.startswith(k)], axis=0) for k in SETS}
    print("\nhybrid (hmm_v6 + seq at a weight): spellF1 macroF1 alive/dead team_acc")
    for i, w in enumerate(HYBRID_WEIGHTS):
        print("  weight {:<5}".format(w) + " | ".join(
            k + " " + " ".join("{:.3f}".format(v) for v in by_set[k][i]) for k in SETS))
    print("\nhybrid - hmm_v6 in spellF1, paired by match, weight chosen on the other two sets:")
    for k in SETS:
        best = int(np.argmax(np.mean([by_set[o][:, 0] for o in SETS if o != k], axis=0)))
        names = [n for n in paths if n.startswith(k)]
        diff = np.array([hyb[n][best, 0] - hyb[n][0, 0] for n in names])
        print("  {:11s} weight {:<5} spellF1 {:.3f}  mean {:+.3f}  standard error {:.3f}  worst "
              "match {:+.3f}  up on {} of {}".format(
                  k, HYBRID_WEIGHTS[best], by_set[k][best, 0], diff.mean(),
                  diff.std(ddof=1) / np.sqrt(len(diff)), diff.min(), int(np.sum(diff > 0)),
                  len(diff)))


if __name__ == "__main__":
    main()
