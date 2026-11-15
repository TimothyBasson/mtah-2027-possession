"""Leave-provider-out sweeps of the control classifier's fit (external/train_control.py).

    python external/sweep_control.py prepare             # decoder inputs per match, once
    python external/sweep_control.py run NAME key=value ...

Each run fits three classifiers, one per held-out provider, decodes the held-out matches with the
shipped decoder at the shipped control weight, and prints folded spellF1 per provider. Results
are appended to data/cache/sweep/results.jsonl, and the held-out posteriors are kept as
data/cache/sweep/NAME.npz so that runs can be averaged afterwards (`blend`).

Keys (defaults are the shipped fit):
    iters=300 lr=0.1 leaves=63 min_leaf=200 l2=1 max_features=1 max_bins=255 depth=0
    stages=300            comma-separated round counts to score from the one fit
    drop=a,b              leave out columns whose names contain any of these
    only=a,b              keep only columns whose names contain any of these
    weight=provider       provider | match | class | match+class | row, and +seen: within a
                          provider, rows are reweighted until the label is independent of whether
                          the ball is seen at the frame (seen_power=1; 0.5 goes half way)
    sub=1                 keep every sub-th training row
    norm=none             none | z (per-match z-score of every column) | rank (per-match quantile)
    seed=0  bag=1         bag > 1 averages that many fits on bootstrap samples of the matches
    sets=idsse,metrica,skillcorner
    within=PROVIDER       not held out: score every 4th match of the provider with a model trained
                          on its other matches (within_train=own) or on those and the other two
                          providers (within_train=all). A ceiling, not a held-out score.
    w=0.2                 control weight(s) in the decoder, comma-separated
"""

import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "external"))
import hmm.decode as dec  # noqa: E402
import score_control as sc  # noqa: E402
import score_hmm as sh  # noqa: E402
import train_control as tc  # noqa: E402
from control import control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402
from hmm.control import STRIDE, hold  # noqa: E402

SWEEP = os.path.join(CACHE, "sweep")
DEFAULTS = dict(iters=300, lr=0.1, leaves=63, min_leaf=200, l2=1.0, max_features=1.0,
                max_bins=255, depth=0, stages="", drop="", only="", weight="provider", sub=1,
                seen_power=1.0, within="", within_train="own",
                norm="none", seed=0, bag=1, sets=",".join(tc.SETS), w="0.2")
STATE = {}


def paths():
    return {os.path.basename(p)[:-4]: p
            for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
            if os.path.basename(p).startswith(tc.SETS)}


def _prepare(args):
    name, path, tabs = args
    out = os.path.join(SWEEP, "ctx_{}.pkl".format(name))
    if os.path.exists(out):
        return name
    speed_t, flight_t, gap_t, width_t = tabs
    d = sh.load(path)
    ball = match_ball(name, d)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    ctx = dict(
        clues=[(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
               (place, dec.place_table()), (sh.speed_cats(d), speed_t),
               (sh.flight_cats(d, ball), flight_t)],
        extra=[(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)],
        segments=[(int(a), int(b) + 1) for a, b in d["halves"]],
        labels=d["labels"], folded=control_labels(d), fps=fps)
    with open(out, "wb") as handle:
        pickle.dump(ctx, handle)
    return name


def prepare():
    os.makedirs(SWEEP, exist_ok=True)
    p = paths()
    tabs = sc.tables(p)
    with Pool(4) as pool:
        for name in pool.imap_unordered(_prepare, [(n, p[n], tabs[n.split("_")[0]]) for n in p]):
            print(name, flush=True)


def _score(args):
    name, post, prior, weights = args
    with open(os.path.join(SWEEP, "ctx_{}.pkl".format(name)), "rb") as handle:
        c = pickle.load(handle)
    ratio = np.log(np.clip(post, 1e-6, 1)) - np.log(prior)
    out = []
    for w in weights:
        pred = dec.decode(c["clues"], dec.transitions(), np.log(dec.START), c["segments"],
                          kickoffs=(), extra=c["extra"], direct=w * ratio if w else None)
        pred[c["labels"] == "X"] = "X"
        out.append(sh.score(c["folded"], pred))
    return name, out


def score(posts, weights, pool):
    """{weight: {set: mean score tuple}} for {match: (held posteriors, prior)}."""
    rows = dict(pool.map(_score, [(n, p, q, weights) for n, (p, q) in posts.items()]))
    out = {}
    for i, w in enumerate(weights):
        out[w] = {k: np.mean([rows[n][i] for n in rows if n.startswith(k)], axis=0)
                  for k in tc.SETS if any(n.startswith(k) for n in rows)}
    return out, rows


def normalise(matches, how):
    """Per-match, label-free rescaling of every column, fitted on both orientations together."""
    out = {}
    for n, m in matches.items():
        both = np.vstack([m[0], m[1]]).astype(np.float64)
        if how == "z":
            mu = np.nanmean(both, axis=0)
            sd = np.nanstd(both, axis=0)
            sd[~(sd > 1e-9)] = 1.0
            f = lambda x: ((x - mu) / sd).astype(np.float32)  # noqa: E731
            out[n] = (f(m[0]), f(m[1])) + m[2:]
        elif how == "rank":
            z = np.empty_like(both, dtype=np.float32)
            for j in range(both.shape[1]):
                col = both[:, j]
                ok = np.isfinite(col)
                r = np.full(len(col), np.nan, np.float32)
                if ok.any():
                    s = np.sort(col[ok])
                    r[ok] = (np.searchsorted(s, col[ok], "left")
                             + np.searchsorted(s, col[ok], "right")) / (2.0 * len(s))
                z[:, j] = r
            out[n] = (z[:len(m[0])], z[len(m[0]):]) + m[2:]
        else:
            raise ValueError(how)
    return out


def rows(matches, names, scheme, sub, counts=None, seen_power=1.0):
    """As train_control.rows, with other weightings. counts: {match: times drawn} for a bag."""
    X, y, w = [], [], []
    seen_col = next(iter(matches.values()))[3].index("d1_h") if "seen" in scheme else None
    for n in names:
        a, b, labels = matches[n][:3]
        g = np.flatnonzero(labels >= 0)[::sub]
        X += [a[g], b[g]]
        yy = np.concatenate([labels[g], tc.SWAP[labels[g]]])
        y.append(yy)
        ww = np.full(len(yy), float(counts[n]) if counts else 1.0)
        w.append(ww)
    prov = np.concatenate([np.full(len(v), tc.SETS.index(n.split("_")[0]))
                           for n, v in zip(names, y)])
    match = np.concatenate([np.full(len(v), i) for i, v in enumerate(y)])
    y, w = np.concatenate(y), np.concatenate(w)
    if "seen" in scheme:    # label independent of ball-seen within each provider
        seen = np.concatenate([np.isfinite(x[:, seen_col]) for x in X])
        dead = y == 2
        for p in np.unique(prov):
            m = prov == p
            share = w[m & dead].sum() / w[m].sum()
            for s in (True, False):
                cell = m & (seen == s)
                cond = w[cell & dead].sum() / max(w[cell].sum(), 1e-9)
                w[cell & dead] *= (share / max(cond, 1e-4)) ** seen_power
                w[cell & ~dead] *= ((1 - share) / max(1 - cond, 1e-4)) ** seen_power
    if "match" in scheme:   # every match the same within its provider
        w = w / np.bincount(match, weights=w)[match]
    if "class" in scheme:   # every class the same within its provider
        for p in np.unique(prov):
            m = prov == p
            w[m] = w[m] / np.bincount(y[m], weights=w[m], minlength=3)[y[m]]
    if scheme != "row":     # every provider the same
        w = w / np.bincount(prov, weights=w)[prov]
    return np.vstack(X), y, w / w.mean()


def fit(X, y, w, c, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=c["iters"], learning_rate=c["lr"], max_leaf_nodes=c["leaves"],
        min_samples_leaf=c["min_leaf"], l2_regularization=c["l2"],
        max_features=c["max_features"], max_bins=c["max_bins"], max_depth=c["depth"] or None,
        early_stopping=False, random_state=seed).fit(X, y, sample_weight=w)


def staged(model, a, b, stages):
    """{rounds: posteriors averaged over the two orientations} from one pass over the stages."""
    out = {}
    for i, (p, q) in enumerate(zip(model.staged_predict_proba(a), model.staged_predict_proba(b))):
        if i + 1 in stages:
            out[i + 1] = ((p + q[:, tc.SWAP]) / 2).astype(np.float32)
    return out


def run(name, c):
    os.makedirs(SWEEP, exist_ok=True)
    matches = tc.load_matches()
    cols = next(iter(matches.values()))[3]
    keep = tc.keep_columns(cols, [d for d in c["drop"].split(",") if d])
    if c["only"]:
        keep = [i for i in keep if any(o in cols[i] for o in c["only"].split(","))]
    kept = [cols[i] for i in keep]
    matches = {n: (m[0][:, keep], m[1][:, keep], m[2], kept) + m[4:] for n, m in matches.items()}
    if c["norm"] != "none":
        matches = normalise(matches, c["norm"])
    stages = [int(s) for s in str(c["stages"]).split(",") if s] or [c["iters"]]
    weights = [float(w) for w in str(c["w"]).split(",")]
    sets = c["sets"].split(",")
    posts = {s: {} for s in stages}
    loss = {s: {} for s in stages}
    t0 = time.time()
    if c["within"]:
        sets = [c["within"]]
    for k in sets:
        train = [n for n in matches if not n.startswith(k)]
        held = [n for n in matches if n.startswith(k)]
        if c["within"]:   # a ceiling: every 4th match of this provider, trained on its others
            own = held
            held = own[::4]
            train = ([n for n in own if n not in held]
                     + (train if c["within_train"] == "all" else []))
        acc = {s: {n: 0.0 for n in held} for s in stages}
        prior = 0.0
        rng = np.random.default_rng(c["seed"])
        for b in range(c["bag"]):
            counts = None
            if c["bag"] > 1:   # bootstrap the matches within each provider
                counts = {n: 0 for n in train}
                for p in tc.SETS:
                    pool_ = [n for n in train if n.startswith(p)]
                    for n in rng.choice(pool_, len(pool_)) if pool_ else []:
                        counts[n] += 1
            used = [n for n in train if counts is None or counts[n]]
            X, y, w = rows(matches, used, c["weight"], c["sub"], counts, c["seen_power"])
            model = fit(X, y, w, c, c["seed"] + b)
            prior = prior + tc.class_shares(y, w) / c["bag"]
            for n in held:
                for s, p in staged(model, matches[n][0], matches[n][1], stages).items():
                    acc[s][n] = acc[s][n] + p / c["bag"]
            del X, y, w, model
        for s in stages:
            ll = []
            for n in held:
                labels, n_frames = matches[n][2], matches[n][5]
                g = labels >= 0
                ll.append(-np.mean(np.log(np.clip(acc[s][n][g, labels[g]], 1e-6, 1))))
                posts[s][n] = (hold(acc[s][n], n_frames), prior)
            loss[s][k] = float(np.mean(ll))
        print("{} held out: {:.0f} s".format(k, time.time() - t0), flush=True)
    with Pool(4) as pool:
        for s in stages:
            tag = name if len(stages) == 1 else "{}@{}".format(name, s)
            np.savez_compressed(
                os.path.join(SWEEP, tag + ".npz"),
                **{n: acc_[0][::STRIDE] for n, acc_ in posts[s].items()},
                **{"frames_" + n: len(acc_[0]) for n, acc_ in posts[s].items()},
                **{"prior_" + n: acc_[1] for n, acc_ in posts[s].items()})
            res, _ = score(posts[s], weights, pool)
            for w in weights:
                report(tag if len(weights) == 1 else "{} w{:g}".format(tag, w), res[w], loss[s],
                       dict(c, stage=s, w=w, columns=len(keep)))


def report(tag, res, loss, config):
    f1 = {k: float(v[0]) for k, v in res.items()}
    mean = float(np.mean(list(f1.values()))) if len(f1) == len(tc.SETS) else float("nan")
    print("{:34s} {}  mean {:.4f} | team_acc {} | logloss {}".format(
        tag, " ".join("{:.4f}".format(f1.get(k, float("nan"))) for k in tc.SETS), mean,
        " ".join("{:.3f}".format(res[k][3]) for k in res),
        " ".join("{:.2f}".format(loss.get(k, float("nan"))) for k in tc.SETS)), flush=True)
    with open(os.path.join(SWEEP, "results.jsonl"), "a") as handle:
        handle.write(json.dumps(dict(
            name=tag, f1=f1, mean=mean, loss=loss, config=config,
            detail={k: [float(x) for x in v] for k, v in res.items()})) + "\n")


def blend(name, sources, weights):
    """Average the saved held-out posteriors of several runs and score the average."""
    files = [dict(np.load(os.path.join(SWEEP, s + ".npz"))) for s in sources]
    posts = {n: (hold(np.mean([f[n] for f in files], axis=0), int(files[0]["frames_" + n])),
                 np.mean([f["prior_" + n] for f in files], axis=0))
             for n in files[0] if not n.startswith(("prior_", "frames_"))}
    with Pool(4) as pool:
        res, _ = score(posts, weights, pool)
    for w in weights:
        report("{} w{:g}".format(name, w), res[w], {}, dict(blend=sources, w=w))


def parse(args):
    c = dict(DEFAULTS)
    for a in args:
        k, v = a.split("=", 1)
        c[k] = type(DEFAULTS[k])(v)
    return c


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "run":
        run(sys.argv[2], parse(sys.argv[3:]))
    elif sys.argv[1] == "blend":
        blend(sys.argv[2], sys.argv[3].split(","),
              [float(w) for w in (sys.argv[4] if len(sys.argv) > 4 else "0.2").split(",")])
