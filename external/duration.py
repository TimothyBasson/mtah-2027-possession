"""Duration-aware spell decoding on the v7 emissions, scored with each provider held out.

    python external/control_features.py
    python external/train_control.py --loso
    python external/duration.py cache        # v7 emissions per match -> data/cache/duration_*.npz
    python external/duration.py baseline     # plain Viterbi: must print score_control.py's v7 row
    python external/duration.py post         # rules after Viterbi (absorb, merge gaps, drop short)
    python external/duration.py hsmm         # explicit-duration (semi-Markov) Viterbi
    python external/duration.py rescore      # spell-level classifier on short spells

Everything that is fitted or chosen (duration histograms, the prior's weight, rule thresholds,
the spell classifier) is fitted on the two providers that are not being scored, against their
folded labels (external/control.py); the provider being scored gives nothing but its tracking.
The emissions are v7's as external/score_control.py builds them: each provider's tables counted on
the others, the control classifier trained without the provider, CONTROL_WEIGHT 0.15.

One leak this shares with every number in score_control.py: when IDSSE is held out, Metrica's and
SkillCorner's emissions (on which the settings are chosen) come from classifiers that saw IDSSE.
The held-out provider's own labels are never read for anything but the final score.

Measured 2026-11-14, folded spellF1, IDSSE / Metrica / SkillCorner (mean), settings chosen without
the provider scored:
    v7, plain Viterbi                    0.609 / 0.798 / 0.428  (0.611)
    rules after Viterbi (post)           0.616 / 0.805 / 0.438  (0.620)
    semi-Markov Viterbi (hsmm)           0.617 / 0.810 / 0.427  (0.618)
    semi-Markov + 2 s absorb             0.613 / 0.810 / 0.436  (0.620)
    spell classifier (rescore)           0.617 / 0.808 / 0.440  (0.622)
    one fixed setting, hsmm lam2 min1.5  0.617 / 0.810 / 0.446  (0.624), picked on all 30 matches
Ceiling of removing spells from v7's path with the labels in hand (not a model): every false
possession under 2 s removed 0.636 / 0.840 / 0.454 (0.643); of any length 0.661 / 0.855 / 0.528
(0.681). So the three ways agree on about +0.01, and no rule on short spells reaches 0.65.
Against the raw labels the rules and the semi-Markov prior cost IDSSE and Metrica 0.03-0.05:
they remove the touch-level spells the folded view folds away.
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
import possession_metrics as pm  # noqa: E402
import hmm.decode as dec  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402
from hmm.model import build_log_emit, viterbi  # noqa: E402

SETS = sh.SETS
H, A, D = 0, 1, 2
STATE = {}


def _init(state):
    STATE.update(state)


def paths():
    return {os.path.basename(p)[:-4]: p
            for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
            if os.path.basename(p).startswith(SETS)}


def cache_path(name):
    return os.path.join(CACHE, "duration_{}.npz".format(name))


# -- the v7 emissions, once per match -------------------------------------------------------------

def _cache_one(name):
    tabs, loso = STATE["tables"], STATE["loso"]
    speed_t, flight_t, gap_t, width_t = tabs[name.split("_")[0]]
    d = sh.load(STATE["paths"][name])
    ball = match_ball(name, d)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t),
             (sh.flight_cats(d, ball), flight_t)]
    emit = build_log_emit([(o, t, dec.WEIGHT) for o, t in clues]
                          + [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)])
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - np.log(loso["prior_" + name])
    emit = emit + dec.CONTROL_WEIGHT * ratio
    np.savez_compressed(
        cache_path(name), emit=emit.astype(np.float32), raw=d["labels"],
        folded=control_labels(d), fps=fps, halves=d["halves"],
        bx=bx.astype(np.float32), by=by.astype(np.float32),
        d_home=d_home.astype(np.float32), d_away=d_away.astype(np.float32),
        post=loso[name].astype(np.float32))
    return name


def build_cache(jobs):
    import score_control as sc
    p = paths()
    todo = [n for n in p if not os.path.exists(cache_path(n))]
    if not todo:
        return
    state = dict(paths=p, tables=sc.tables(p),
                 loso=dict(np.load(os.path.join(CACHE, "control_loso.npz"))))
    with Pool(jobs, initializer=_init, initargs=(state,)) as pool:
        for n in pool.imap_unordered(_cache_one, todo):
            print("cached", n, flush=True)


def load_all():
    out = {}
    for n in paths():
        z = dict(np.load(cache_path(n)))
        z["fps"] = int(z["fps"])
        z["emit"] = z["emit"].astype(np.float64)
        z["segments"] = [(int(a), int(b) + 1) for a, b in z["halves"]]
        z["gt"] = [pm.label_codes(z["raw"]), pm.label_codes(z["folded"])]
        z["runs"] = [pm.valid_runs(g) for g in z["gt"]]
        z["gt_spells"] = [pm.possession_spells(g, r) for g, r in zip(z["gt"], z["runs"])]
        out[n] = z
    return out


# -- scoring --------------------------------------------------------------------------------------

def overlaps(gt_spells, pred_spells):
    """pm.spell_overlaps without the all-pairs loop: both lists are sorted and disjoint."""
    out, j0 = [], 0
    for i, (ga, gb, gk) in enumerate(gt_spells):
        while j0 < len(pred_spells) and pred_spells[j0][1] <= ga:
            j0 += 1
        j = j0
        while j < len(pred_spells) and pred_spells[j][0] < gb:
            pa, pb, pk = pred_spells[j]
            if pk == gk:
                out.append(((min(gb, pb) - max(ga, pa)) / float(max(gb, pb) - min(ga, pa)), i, j))
            j += 1
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def spell_f1(m, pred, view=1):
    """(mean spellF1, predicted possessions) of string labels `pred` against match m's labels:
    view 0 as they are, view 1 folded. Same number as possession_metrics.evaluate."""
    codes = pm.label_codes(np.where(m["raw"] == "X", "X", pred))
    runs, gt_spells = m["runs"][view], m["gt_spells"][view]
    pred_spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    ov = overlaps(gt_spells, pred_spells)
    f1s = []
    for thr in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, len(gt_spells), len(pred_spells), thr)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)), len(pred_spells)


def by_set(scores):
    """{set: mean over its matches} from {match: score}."""
    return {k: float(np.mean([v for n, v in scores.items() if n.startswith(k)])) for k in SETS}


def line(tag, per_set, base=None):
    vals = [per_set[k] for k in SETS]
    s = "{:34s} {:.3f} / {:.3f} / {:.3f}   mean {:.3f}".format(tag, *vals, np.mean(vals))
    if base is not None:
        s += "   vs v7 " + " ".join("{:+.3f}".format(per_set[k] - base[k]) for k in SETS)
    return s


# -- decoders -------------------------------------------------------------------------------------

def decode_viterbi(m, trans=None):
    labels = np.full(len(m["emit"]), "X", dtype="<U1")
    trans = dec.transitions() if trans is None else trans
    for lo, hi in m["segments"]:
        labels[lo:hi] = dec.STATES[viterbi(m["emit"][lo:hi], trans, np.log(dec.START))]
    return labels


def runs_of(seg):
    starts = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]
    return starts, np.r_[starts[1:], len(seg)]


def postprocess(m, labels, absorb=0.0, gap=0.0, short=0.0, short_to="emit"):
    """Rules applied to a decoded path, each inside a half, repeated until nothing changes.

    absorb  a team spell under `absorb` s with the other team on both sides goes to that team
    gap     a stoppage under `gap` s with the same team on both sides goes to that team
    short   any other team spell under `short` s goes to a neighbour: the one whose label the
            emissions prefer over the spell's frames ("emit"), or always D if it touches one ("dead")
    """
    fps, out = m["fps"], labels.copy()
    for lo, hi in m["segments"]:
        seg, emit = out[lo:hi], m["emit"][lo:hi]
        cum = np.vstack([np.zeros(3), np.cumsum(emit, axis=0)])
        for _ in range(20):
            starts, stops = runs_of(seg)
            changed = False
            for i in range(len(starts)):
                a, b, k = starts[i], stops[i], seg[starts[i]]
                before = seg[a - 1] if a > 0 else ""
                after = seg[b] if b < len(seg) else ""
                n = (b - a) / fps
                new = None
                if k == "D":
                    if n < gap and before == after and before in ("H", "A"):
                        new = before
                elif before and after:
                    if before == after and before != "D":
                        if n < absorb:
                            new = before
                    elif n < short:
                        if short_to == "dead" and "D" in (before, after):
                            new = "D"
                        else:
                            sums = cum[b] - cum[a]
                            idx = {"H": 0, "A": 1, "D": 2}
                            new = before if sums[idx[before]] >= sums[idx[after]] else after
                if new is not None and new != k:
                    seg[a:b] = new
                    changed = True
            if not changed:
                break
    return out


def fit_durations(matches, names, max_seconds, step, floor=1e-4, smooth=2.0):
    """log p(duration | possession) and | stoppage, on a grid of `step` seconds up to max_seconds,
    from the folded labels of `names`, each provider counting once. Returns (log_pos, log_dead,
    exit shares): the histograms are smoothed with a kernel `smooth` s wide on a log-ish scale by
    simple Gaussian blur in bins, floored, and normalised per bin of `step` seconds."""
    n_bins = int(round(max_seconds / step))
    hist = {k: np.zeros((2, n_bins)) for k in SETS}
    exits = {k: np.zeros(2) for k in SETS}
    for n in names:
        m, k = matches[n], n.split("_")[0]
        for lo, hi in m["segments"]:
            seg = m["folded"][lo:hi]
            seg = seg[seg != "X"] if np.any(seg == "X") else seg
            starts, stops = runs_of(seg)
            for i in range(1, len(starts) - 1):  # whole spells only
                kind = seg[starts[i]]
                if kind == "X":
                    continue
                b = min(int((stops[i] - starts[i]) / m["fps"] / step), n_bins - 1)
                hist[k][int(kind == "D"), b] += 1
                if kind != "D":
                    exits[k][int(seg[stops[i]] == "D")] += 1
    used = [k for k in SETS if hist[k].sum()]
    pmf = np.mean([hist[k] / hist[k].sum(axis=1, keepdims=True) for k in used], axis=0)
    width = max(smooth / step, 1e-6)
    x = np.arange(-int(4 * width) - 1, int(4 * width) + 2)
    kernel = np.exp(-0.5 * (x / width) ** 2)
    kernel /= kernel.sum()
    pmf = np.array([np.convolve(row, kernel, mode="same") for row in pmf]) + floor / n_bins
    pmf /= pmf.sum(axis=1, keepdims=True)
    to_dead = float(np.mean([exits[k][1] / exits[k].sum() for k in used]))
    return np.log(pmf[0]), np.log(pmf[1]), to_dead


def decode_hsmm(m, log_pos, log_dead, to_dead, step, lam=1.0, min_pos=0.0):
    """Explicit-duration Viterbi. A path is a chain of spells; a spell of state j lasting d costs
    lam * log p_j(d) plus the transition into it, and its frames' emissions. The emissions are
    summed into blocks of `step` seconds first, so a boundary sits on a block edge. A spell can run
    past the longest duration by chaining into the same state (j -> j is allowed at no cost, only
    from a spell of the longest length). min_pos rules out possessions under that many seconds."""
    fps = m["fps"]
    block = max(int(round(step * fps)), 1)
    n_d = len(log_pos)
    logdur = lam * np.array([log_pos, log_pos, log_dead])            # (3, n_d), d = 1..n_d blocks
    if min_pos > 0:
        logdur[:2, :max(int(round(min_pos / step)) - 1, 0)] = -1e9
    logA = np.log(np.array([[1e-300, 1 - to_dead, to_dead],
                            [1 - to_dead, 1e-300, to_dead],
                            [0.5, 0.5, 1e-300]]))
    labels = np.full(len(m["emit"]), "X", dtype="<U1")
    for lo, hi in m["segments"]:
        emit = m["emit"][lo:hi]
        T = int(np.ceil(len(emit) / block))
        pad = np.zeros((T * block, 3))
        pad[:len(emit)] = emit
        cum = np.vstack([np.zeros(3), np.cumsum(pad.reshape(T, block, 3).sum(axis=1), axis=0)])
        # delta[t, j]: best score of a chain whose last spell is j and ends after block t (1-based)
        delta = np.full((T + 1, 3), -np.inf)
        best_d = np.zeros((T + 1, 3), dtype=np.int32)
        best_i = np.zeros((T + 1, 3), dtype=np.int8)
        # enter[t, j]: best way into a spell of j that starts after block t, minus cum[t, j]
        enter = np.full((T + 1, 3), -np.inf)
        enter_i = np.zeros((T + 1, 3), dtype=np.int8)
        enter[0] = np.log(dec.START) - cum[0]
        enter_i[0] = -1
        rev = logdur[:, ::-1]                                          # rev[:, -d] = logdur[:, d-1]
        for t in range(1, T + 1):
            a = max(t - n_d, 0)
            cand = enter[a:t].T + rev[:, n_d - (t - a):]               # (3, t-a): starts a..t-1
            k = np.argmax(cand, axis=1)
            delta[t] = cand[np.arange(3), k] + cum[t]
            best_d[t] = (t - a) - k
            best_i[t] = enter_i[a + k, np.arange(3)]
            into = delta[t][:, None] + logA                            # from i (rows) to j (cols)
            # a spell of the longest length may continue in the same state
            cont = np.where(best_d[t] == n_d, delta[t], -np.inf)
            into[np.arange(3), np.arange(3)] = cont
            enter_i[t] = np.argmax(into, axis=0)
            enter[t] = into[enter_i[t], np.arange(3)] - cum[t]
        path = np.empty(T, dtype=np.int8)
        t, j = T, int(np.argmax(delta[T]))
        while t > 0:
            d, i = int(best_d[t, j]), int(best_i[t, j])
            path[t - d:t] = j
            t, j = t - d, i
        labels[lo:hi] = dec.STATES[np.repeat(path, block)[:hi - lo]]
    return labels


# -- experiments ----------------------------------------------------------------------------------

def _eval(args):
    """(match, {config key: (folded spellF1, raw spellF1, possessions)}) for a list of configs."""
    name = args
    m, configs = STATE["matches"][name], STATE["configs"][name]
    out, memo = {}, {}
    for key, fn in configs:
        pred = fn(m, memo)
        f, n = spell_f1(m, pred, 1)
        out[key] = (f, spell_f1(m, pred, 0)[0], n)
    return name, out


def run_configs(matches, jobs_by_match, jobs):
    """jobs_by_match: {match: [(key, fn(m, memo) -> labels)]}. Returns {key: {match: scores}}."""
    # the configs are closures, which do not pickle: the workers inherit them by fork
    STATE.update(matches=matches, configs=jobs_by_match)
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        res = dict(pool.map(_eval, list(jobs_by_match), chunksize=1))
    out = {}
    for n, r in res.items():
        for key, v in r.items():
            out.setdefault(key, {})[n] = v
    return out


def pick_held_out(results, keys, base_key, tag):
    """For each provider: the key with the best mean folded spellF1 over the other two providers,
    then that key's score on the provider. Prints the held-out row and every key's own row."""
    fold = {key: by_set({n: v[0] for n, v in results[key].items()}) for key in keys}
    raw = {key: by_set({n: v[1] for n, v in results[key].items()}) for key in keys}
    count = {key: by_set({n: v[2] for n, v in results[key].items()}) for key in keys}
    base = fold[base_key]
    chosen = {k: max(keys, key=lambda key: np.mean([fold[key][o] for o in SETS if o != k]))
              for k in SETS}
    print(line("v7 (plain Viterbi)", base))
    print(line(tag + ", chosen without the provider", {k: fold[chosen[k]][k] for k in SETS}, base))
    print("   chosen: " + "; ".join("{} -> {}".format(k, chosen[k]) for k in SETS))
    print("   raw labels: " + line("", {k: raw[chosen[k]][k] for k in SETS}, raw[base_key]).strip())
    print("   possessions: " + " / ".join("{:.0f} (v7 {:.0f})".format(
        count[chosen[k]][k], count[base_key][k]) for k in SETS))
    return fold, chosen


def _base(m, memo):
    if "base" not in memo:
        memo["base"] = decode_viterbi(m)
    return memo["base"]


def cmd_baseline(matches, a):
    res = run_configs(matches, {n: [("v7", _base)] for n in matches}, a.jobs)
    for v, tag in ((0, "folded"), (1, "raw")):
        print(line("v7 " + tag, by_set({n: s[v] for n, s in res["v7"].items()})))
    m = next(iter(matches.values()))
    pred = decode_viterbi(m)
    ref = sh.score(m["folded"], np.where(m["raw"] == "X", "X", pred))[0]
    print("check against possession_metrics.evaluate on one match: {:.6f} vs {:.6f}".format(
        spell_f1(m, pred)[0], ref))


def _post(**kw):
    return lambda m, memo: postprocess(m, _base(m, memo), **kw)


def cmd_post(matches, a):
    configs = [("v7", _base)]
    for absorb in (0, 1, 1.5, 2, 2.5, 3):
        for gap in (0, 0.5, 1, 2, 3):
            for short in (0, 1, 2, 3):
                for to in (("emit", "dead") if short else ("emit",)):
                    key = "absorb{:g} gap{:g} short{:g}{}".format(
                        absorb, gap, short, "-" + to if short else "")
                    configs.append((key, _post(absorb=absorb, gap=gap, short=short, short_to=to)))
    res = run_configs(matches, {n: configs for n in matches}, a.jobs)
    keys = [k for k, _ in configs]
    fold, _ = pick_held_out(res, keys, "v7", "rules after Viterbi")
    print("\nsingle rules and the main thread's 2 s absorb, same setting on every provider:")
    for key in ("absorb2 gap0 short0", "absorb1.5 gap0 short0", "absorb3 gap0 short0",
                "absorb0 gap1 short0", "absorb0 gap2 short0", "absorb0 gap3 short0",
                "absorb0 gap0 short1-emit", "absorb0 gap0 short2-emit", "absorb0 gap0 short2-dead",
                "absorb2 gap1 short0", "absorb2 gap0 short1-emit", "absorb2 gap0 short2-emit"):
        print(line(key, fold[key], fold["v7"]))
    best = sorted(keys, key=lambda k: -np.mean(list(fold[k].values())))[:5]
    print("\nbest five by the mean over all three (in-sample, for reference only):")
    for key in best:
        print(line(key, fold[key], fold["v7"]))


def cmd_hsmm(matches, a):
    step, max_s = a.step, a.max_seconds
    lams = [float(v) for v in a.lams.split(",")]
    mins = [float(v) for v in a.min_pos.split(",")]
    fits = {k: fit_durations(matches, [n for n in matches if not n.startswith(k)], max_s, step)
            for k in SETS}
    for k in SETS:
        lp, ld, td = fits[k]
        grid = (np.arange(len(lp)) + 0.5) * step
        print("{} held out: possession mean {:.1f} s, under 2 s {:.1%}; stoppage mean {:.1f} s; "
              "exits to dead {:.2f}".format(k, np.sum(np.exp(lp) * grid),
                                            np.exp(lp)[:int(2 / step)].sum(),
                                            np.sum(np.exp(ld) * grid), td))

    # Every match is decoded with the durations fitted without its own provider, when it is
    # scored and when its score helps choose the settings for another provider alike.
    configs = [("v7", _base), ("v7 absorb2", _post(absorb=2.0))]
    for lam in lams:
        for mp in mins:
            for absorb in (0.0, 2.0):
                def fn(m, memo, lam=lam, mp=mp, absorb=absorb):
                    key = ("hsmm", lam, mp)
                    if key not in memo:
                        memo[key] = decode_hsmm(m, *fits[m["set"]], step, lam, mp)
                    return postprocess(m, memo[key], absorb=absorb) if absorb else memo[key]
                configs.append(("hsmm lam{:g} min{:g}{}".format(
                    lam, mp, " absorb2" if absorb else ""), fn))
    for n, m in matches.items():
        m["set"] = n.split("_")[0]
    res = run_configs(matches, {n: configs for n in matches}, a.jobs)
    keys = [k for k, _ in configs]
    pick_held_out(res, [k for k in keys if "absorb" not in k], "v7", "semi-Markov Viterbi")
    fold, _ = pick_held_out(res, keys, "v7", "semi-Markov + 2 s absorb")
    print("\nevery setting, same on all providers:")
    for key in keys:
        print(line(key, fold[key], fold["v7"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["cache", "baseline", "post", "hsmm", "rescore"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.2, help="hsmm block length, seconds")
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--lams", default="0.5,1,2,3")
    ap.add_argument("--min-pos", default="0,1,2")
    a = ap.parse_args()
    build_cache(a.jobs)
    if a.what == "cache":
        return
    matches = load_all()
    if a.what == "rescore":
        import rescore
        return rescore.run(matches, a)
    {"baseline": cmd_baseline, "post": cmd_post, "hsmm": cmd_hsmm}[a.what](matches, a)


if __name__ == "__main__":
    main()
