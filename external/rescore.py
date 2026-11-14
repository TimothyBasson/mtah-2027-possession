"""Spell-level rescoring: a classifier says which decoded possessions are real, the rest are merged
into a neighbour. After Link and Hoernig (2017), who settle individual ball possession with rules
on the spell (duration, ball speed, opponents near); here the rule is learned.

    python external/duration.py rescore

Candidates are the possessions of a Viterbi path, v7's or one decoded with a shorter mean
possession so that it over-segments. Each candidate gets features of the whole spell (below) and
the label "matches a possession of the folded labels, same team, IoU >= 0.3, one to one". A
gradient-boosted classifier (scikit-learn) is trained on the candidates of the two providers that
are not being scored. Candidates under `max_dur` seconds whose probability is under `threshold`
are given to a neighbour: the label on both sides if they agree, else the side the emissions
prefer over the spell's frames. The three settings (candidate source, max_dur, threshold) are
chosen on the two training providers, each scored by a classifier trained on the other one alone.
"""

import multiprocessing

import numpy as np

import duration as du
import possession_metrics as pm
import hmm.decode as dec
from hmm.model import rate_log_trans

SOURCES = (12.0, 6.0, 3.0)        # mean possession seconds of the path the candidates come from
MAX_DURS = (2.0, 4.0, 8.0, 1e9)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)
IDX = {"H": 0, "A": 1, "D": 2}
NAMES = ["seconds", "margin_other", "margin_dead", "sum_other", "sum_dead", "post_own",
         "post_other", "post_dead", "ball_seen", "speed_mean", "speed_max", "speed_std",
         "accel_mean", "d_own_mean", "d_own_min", "d_opp_mean", "d_opp_min", "own_nearer",
         "prev_other", "prev_dead", "next_other", "next_dead", "prev_seconds", "next_seconds",
         "margin_prev", "margin_next"]


def candidates(m, source):
    """(labels, [(start, stop)] of its possessions, features, real) for one match."""
    fps = m["fps"]
    trans = rate_log_trans(source * dec.FPS, dec.STOPPAGE_SECONDS * dec.FPS, dec.TO_DEAD_SHARE)
    labels = du.decode_viterbi(m, trans)
    with np.errstate(invalid="ignore"):
        speed = np.r_[np.nan, np.hypot(np.diff(m["bx"]), np.diff(m["by"]))] * fps
        accel = np.abs(np.r_[np.nan, np.diff(speed)]) * fps
    spans, rows = [], []
    for lo, hi in m["segments"]:
        seg = labels[lo:hi]
        starts, stops = du.runs_of(seg)
        cum = np.vstack([np.zeros(3), np.cumsum(m["emit"][lo:hi], axis=0)])
        for i in range(len(starts)):
            a, b, k = int(starts[i]), int(stops[i]), seg[starts[i]]
            if k == "D":
                continue
            own, other = IDX[k], 1 - IDX[k]
            sums = cum[b] - cum[a]
            n = b - a
            sl = slice(lo + a, lo + b)
            post = m["post"][sl].mean(axis=0)
            seen = np.isfinite(m["bx"][sl])
            d_own = (m["d_home"], m["d_away"])[own][sl][seen]
            d_opp = (m["d_home"], m["d_away"])[other][sl][seen]
            sp, ac = speed[sl], accel[sl]
            sp, ac = sp[np.isfinite(sp)], ac[np.isfinite(ac)]
            stat = lambda v, f: float(f(v)) if len(v) else np.nan  # noqa: E731
            prev = seg[a - 1] if i > 0 else ""
            nxt = seg[b] if i + 1 < len(starts) else ""
            rows.append([
                n / fps, (sums[own] - sums[other]) / n, (sums[own] - sums[2]) / n,
                (sums[own] - sums[other]) / fps, (sums[own] - sums[2]) / fps,
                post[own], post[other], post[2], seen.mean(),
                stat(sp, np.mean), stat(sp, np.max), stat(sp, np.std), stat(ac, np.mean),
                stat(d_own, np.nanmean), stat(d_own, np.nanmin), stat(d_opp, np.nanmean),
                stat(d_opp, np.nanmin),
                stat(d_own < d_opp, np.mean),
                prev not in ("", "D"), prev == "D", nxt not in ("", "D"), nxt == "D",
                (a - starts[i - 1]) / fps if i > 0 else np.nan,
                (stops[i + 1] - b) / fps if i + 1 < len(starts) else np.nan,
                (sums[own] - sums[IDX[prev]]) / fps if prev else np.nan,
                (sums[own] - sums[IDX[nxt]]) / fps if nxt else np.nan])
            spans.append((lo + a, lo + b))
    X = np.array(rows, dtype=float)
    # real: matched one to one with a possession of the folded labels at IoU >= 0.3
    codes = pm.label_codes(np.where(m["raw"] == "X", "X", labels))
    runs, gt_spells = m["runs"][1], m["gt_spells"][1]
    pred_spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    matched = set()
    done_gt = set()
    for iou, gi, pi in du.overlaps(gt_spells, pred_spells):
        if iou < 0.3:
            break
        if gi in done_gt or pi in matched:
            continue
        done_gt.add(gi)
        matched.add(pi)
    # the scorer's spells are cut at ungraded frames; a candidate is real if a matched scorer
    # spell covers its midpoint
    mids = np.array([(a + b) // 2 for a, b in spans])
    real = np.zeros(len(spans), dtype=bool)
    for pi in matched:
        a, b, _ = pred_spells[pi]
        real |= (mids >= a) & (mids < b)
    return labels, spans, X, real


def apply(m, labels, spans, X, prob, max_dur, threshold):
    """Give the candidates under max_dur seconds with prob under threshold to a neighbour."""
    out = labels.copy()
    cum = np.vstack([np.zeros(3), np.cumsum(m["emit"], axis=0)])
    edges = {lo for lo, _ in m["segments"]} | {hi for _, hi in m["segments"]}
    for j in np.argsort(prob):
        if prob[j] >= threshold:
            break
        if X[j, 0] >= max_dur:
            continue
        a, b = spans[j]
        before = out[a - 1] if a not in edges else ""
        after = out[b] if b not in edges else ""
        if not before and not after:
            continue
        if before and after and before != after:
            sums = cum[b] - cum[a]
            new = before if sums[IDX[before]] >= sums[IDX[after]] else after
        else:
            new = before or after
        out[a:b] = new
    return out


def fit(Xs, ys, sets):
    """One classifier on the candidates of several providers, each provider weighing the same."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = np.vstack(Xs), np.concatenate(ys)
    w = np.concatenate([np.full(len(x), 1.0 / sum(len(q) for q, t in zip(Xs, sets) if t == s))
                        for x, s in zip(Xs, sets)])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0).fit(
                                              X, y, sample_weight=w * len(y))


def _cands(args):
    name, source = args
    return (name, source), candidates(du.STATE["matches"][name], source)


def run(matches, a):
    du.STATE.update(matches=matches)
    with multiprocessing.get_context("fork").Pool(a.jobs) as pool:
        cands = dict(pool.map(_cands, [(n, s) for n in matches for s in SOURCES], chunksize=1))
    set_of = {n: n.split("_")[0] for n in matches}
    base = du.by_set({n: du.spell_f1(m, du.decode_viterbi(m))[0] for n, m in matches.items()})

    def probs(train_sets, source):
        """{match: prob} for every match outside train_sets, from a classifier fitted on them."""
        names = [n for n in matches if set_of[n] in train_sets]
        model = fit([cands[n, source][2] for n in names], [cands[n, source][3] for n in names],
                    [set_of[n] for n in names])
        return {n: model.predict_proba(cands[n, source][2])[:, 1]
                for n in matches if set_of[n] not in train_sets}, model

    def score(names, prob, source, max_dur, threshold):
        s = {}
        for n in names:
            labels, spans, X, _ = cands[n, source]
            s[n] = du.spell_f1(matches[n], apply(matches[n], labels, spans, X, prob[n], max_dur,
                                                 threshold))
        return s

    grid = [(s, d, t) for s in SOURCES for d in MAX_DURS for t in THRESHOLDS]
    single = {(k, s): probs({k}, s)[0] for k in du.SETS for s in SOURCES}
    held, chosen, counts, raw = {}, {}, {}, {}
    for k in du.SETS:
        others = [o for o in du.SETS if o != k]
        best, best_val = None, -1
        for s, d, t in grid:
            # each training provider scored by the classifier trained on the other one alone
            val = np.mean([np.mean([v[0] for v in score(
                [n for n in matches if set_of[n] == o], single[p, s], s, d, t).values()])
                for o, p in (others, others[::-1])])
            if val > best_val:
                best, best_val = (s, d, t), val
        s, d, t = best
        prob, model = probs(set(others), s)
        names = [n for n in matches if set_of[n] == k]
        res = score(names, prob, s, d, t)
        held[k] = float(np.mean([v[0] for v in res.values()]))
        counts[k] = float(np.mean([v[1] for v in res.values()]))
        chosen[k] = "source {:g} s, under {:g} s, p < {:g} (inner {:.3f})".format(s, d, t, best_val)
        real = np.concatenate([cands[n, s][3] for n in names])
        p_all = np.concatenate([prob[n] for n in names])
        print("{} held out: {} candidates, {:.0%} real; mean p on real {:.2f}, on false {:.2f}"
              .format(k, len(real), real.mean(), p_all[real].mean(), p_all[~real].mean()))
    print(du.line("v7 (plain Viterbi)", base))
    print(du.line("spell classifier, chosen without", held, base))
    for k in du.SETS:
        print("   {}: {}; possessions {:.0f}".format(k, chosen[k], counts[k]))
    print("\nfixed settings on every provider (classifier still trained without the provider):")
    full = {(k, s): probs(set(du.SETS) - {k}, s)[0] for k in du.SETS for s in SOURCES}
    for s, d, t in grid:
        if d not in (4.0, 1e9) or t not in (0.3, 0.5):
            continue
        row = {k: float(np.mean([v[0] for v in score(
            [n for n in matches if set_of[n] == k], full[k, s], s, d, t).values()]))
            for k in du.SETS}
        print(du.line("source {:g} under {:g} p<{:g}".format(s, d, t), row, base))
