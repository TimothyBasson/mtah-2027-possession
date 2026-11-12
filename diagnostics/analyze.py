"""Where a possession model loses spellF1, match by match, from predictions saved by predict.py.

    python diagnostics/analyze.py main        # -> data/cache/diag/main.json

For every outside match: the score, the score with one kind of mistake repaired from the truth
(everything else left as the model had it), recall by possession length and by what starts and
ends the possession, and how early or late the model's turnovers and stoppages are.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from lib import FPS, pm  # noqa: E402

DUR_BINS = [0, 1, 2, 3, 5, 8, 12, 20, 1e9]
DUR_NAMES = ["<1s", "1-2s", "2-3s", "3-5s", "5-8s", "8-12s", "12-20s", ">20s"]
SHIFTS = [-50, -37, -25, -18, -12, -6, 0, 6, 12, 18, 25, 37, 50]


def absorb(labels, min_len):
    """possession/model.py's absorb_short, run-length based."""
    import itertools
    runs = [[k, len(list(grp))] for k, grp in itertools.groupby(labels.tolist())]
    while len(runs) > 1:
        short = [i for i, (k, n) in enumerate(runs) if n < min_len.get(k, 0)]
        if not short:
            break
        i = min(short, key=lambda i: runs[i][1])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: runs[j][1])
        runs[j][1] += runs.pop(i)[1]
        if 0 < i < len(runs) and runs[i - 1][0] == runs[i][0]:
            runs[i - 1][1] += runs.pop(i)[1]
    return np.array([k for k, n in runs for _ in range(n)], dtype=labels.dtype)


def truth_after_absorb(g, min_team, min_dead):
    """The truth passed through the model's own clean-up: the best score that clean-up allows."""
    gc = pm.label_codes(g)
    alive, team = np.isin(g, ["H", "A"]), lib.fill_team(g)
    for a, b in pm.valid_runs(gc):
        alive[a:b] = absorb(alive[a:b], {True: min_team * FPS, False: min_dead * FPS})
        team[a:b] = absorb(team[a:b], {"H": min_team * FPS, "A": min_team * FPS})
    return lib.compose(alive, team, g)


def nearest_offsets(true_at, pred_at, tol):
    """For each true event the signed offset (pred - true, frames) of the nearest predicted one
    within tol, NaN if none; and how many predicted events are nobody's nearest."""
    pred_at = np.asarray(pred_at)
    off = np.full(len(true_at), np.nan)
    used = np.zeros(len(pred_at), bool)
    if len(pred_at):
        for i, t in enumerate(true_at):
            j = int(np.argmin(np.abs(pred_at - t)))
            if abs(pred_at[j] - t) <= tol:
                off[i] = pred_at[j] - t
                used[j] = True
    return off, int((~used).sum())


def one(args):
    tag, name = args
    g = lib.truth(name)
    z = lib.preds(tag, name)
    pred, team = z["pred_heldout"], z["team"]
    graded = g != "X"
    L, M = np.isin(g, ["H", "A"]), np.isin(pred, ["H", "A"])
    Gt = lib.fill_team(g)
    out = {"name": name, "provider": lib.provider(name)}

    r = pm.evaluate(pm.label_codes(g), pm.label_codes(pred), bootstrap_n=300)
    out["spellF1"] = r["spell_f1_mean"]
    out["boot"] = [r["spell_bootstrap_low"], r["spell_bootstrap_high"]]
    out["per_iou"] = {str(k): [v["precision"], v["recall"], v["f1"]] for k, v in r["spell_f1"].items()}
    out["n_gt"], out["n_pred"] = r["n_gt_spells"], r["n_pred_spells"]
    out["macroF1"], out["alive_dead"], out["team_acc"] = r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]
    out["insample"] = lib.spell_f1(g, z["pred_insample"])
    out["graded_min"] = float(graded.sum() / FPS / 60)
    out["team_acc_live"] = float((team[L] == g[L]).mean())  # team call on every truly live frame
    out["dead_share_true"], out["dead_share_pred"] = float((g[graded] == "D").mean()), float((pred[graded] == "D").mean())

    # -- repairs -------------------------------------------------------------------------------
    S = lambda p: lib.spell_f1(g, p)  # noqa: E731
    base = S(lib.compose(M, team, g))
    assert abs(base - out["spellF1"]) < 1e-9, (base, out["spellF1"])
    rep = {}
    rep["true_dead__model_team"] = S(lib.compose(L, team, g))
    rep["model_dead__true_team"] = S(lib.compose(M, Gt, g))
    # true boundaries, team by the model's majority inside each true possession
    maj = Gt.copy()
    for a, b, t in lib.spells(g):
        maj[a:b] = "H" if (team[a:b] == "H").mean() >= 0.5 else "A"
    rep["true_segments__model_majority_team"] = S(lib.compose(L, maj, g))
    rep["truth_after_absorb_2s_3s"] = S(truth_after_absorb(g, 2.0, 3.0))
    rep["truth_after_absorb_1s_1s"] = S(truth_after_absorb(g, 1.0, 1.0))
    rep["truth_after_absorb_team2s_only"] = S(truth_after_absorb(g, 2.0, 0.0))
    rep["truth_after_absorb_dead3s_only"] = S(truth_after_absorb(g, 0.0, 3.0))

    tc = lib.team_error_classes(g, team)
    dc = lib.dead_error_classes(g, M)
    frames = {}
    for k, m in tc.items():
        frames[k] = int(m.sum())
        t2 = np.where(m, Gt, team)
        rep["fix_team_" + k] = S(lib.compose(M, t2, g))                    # alone, model's stoppages
        rep["fix_team_" + k + "__true_dead"] = S(lib.compose(L, t2, g))    # alone, true stoppages
        t3 = np.where(m, team, Gt)                                          # everything but this
        rep["only_team_" + k + "__true_dead"] = S(lib.compose(L, t3, g))
    for k, m in dc.items():
        frames[k] = int(m.sum())
        rep["fix_" + k] = S(lib.compose(np.where(m, L, M), team, g))
        rep["fix_" + k + "__true_team"] = S(lib.compose(np.where(m, L, M), Gt, g))
        rep["only_" + k + "__true_team"] = S(lib.compose(np.where(m, M, L), Gt, g))
    groups = {
        "team_timing": ["late_start_turnover", "late_start_restart", "early_end_turnover", "early_end_stoppage"],
        "team_timing_at_turnovers": ["late_start_turnover", "early_end_turnover"],
        "team_timing_at_stoppages": ["late_start_restart", "early_end_stoppage"],
        "team_whole": ["whole_under2s", "whole_2to5s", "whole_over5s"],
        "team_interior": ["interior"],
    }
    for name_, ks in groups.items():
        m = np.any([tc[k] for k in ks], 0)
        rep["fix_group_" + name_] = S(lib.compose(M, np.where(m, Gt, team), g))
        rep["fix_group_" + name_ + "__true_dead"] = S(lib.compose(L, np.where(m, Gt, team), g))
    dgroups = {"false_dead": ["false_dead_whole", "false_dead_edge", "false_dead_interior"],
               "missed_dead": ["missed_dead_whole", "missed_dead_edge", "missed_dead_interior"],
               "dead_timing": ["false_dead_edge", "missed_dead_edge"]}
    for name_, ks in dgroups.items():
        m = np.any([dc[k] for k in ks], 0)
        rep["fix_group_" + name_] = S(lib.compose(np.where(m, L, M), team, g))
        rep["fix_group_" + name_ + "__true_team"] = S(lib.compose(np.where(m, L, M), Gt, g))
    out["repairs"], out["error_frames"] = rep, frames
    out["live_frames"], out["graded_frames"] = int(L.sum()), int(graded.sum())

    # -- recall by length and by context --------------------------------------------------------
    gs, ps, gm, pmt, best = lib.matching(g, pred)
    gd = np.array([(b - a) / FPS for a, b, _ in gs])
    pd_ = np.array([(b - a) / FPS for a, b, _ in ps])
    out["recall_by_dur"] = [[int(((gd >= lo) & (gd < hi)).sum()), float(gm[(gd >= lo) & (gd < hi)].sum())]
                            for lo, hi in zip(DUR_BINS, DUR_BINS[1:])]
    out["precision_by_dur"] = [[int(((pd_ >= lo) & (pd_ < hi)).sum()), float(pmt[(pd_ >= lo) & (pd_ < hi)].sum())]
                               for lo, hi in zip(DUR_BINS, DUR_BINS[1:])]
    n = len(g)
    ctx = {}
    for (a, b, _), m_ in zip(gs, gm):
        s = "turnover" if a > 0 and g[a - 1] in "HA" else "restart" if a > 0 and g[a - 1] == "D" else "cut"
        e = "turnover" if b < n and g[b] in "HA" else "stoppage" if b < n and g[b] == "D" else "cut"
        c = ctx.setdefault(s + ">" + e, [0, 0.0])
        c[0] += 1
        c[1] += float(m_)
    out["recall_by_context"] = ctx
    # what the unmatched true possessions look like in the prediction (at IoU 0.5)
    why = {}
    for (a, b, t), m_, bi in zip(gs, gm, best):
        if bi >= 0.5:
            continue
        seg = pred[a:b]
        right, wrong, dead = (seg == pm.LABELS[t]).mean(), (np.isin(seg, ["H", "A"]) & (seg != pm.LABELS[t])).mean(), (seg == "D").mean()
        if right < 0.5:
            k = "mostly_wrong_team" if wrong >= dead else "mostly_called_dead"
        else:
            n_right = len(lib.runs_of(seg == pm.LABELS[t]))
            if n_right > 1:
                k = "split_in_pieces"
            else:
                # one piece, at least half covered: the piece runs on beyond the possession, or falls short
                s0, e0 = lib.runs_of(seg == pm.LABELS[t])[0]
                lo, hi = a + s0, a + e0
                while lo > 0 and pred[lo - 1] == pm.LABELS[t]:
                    lo -= 1
                while hi < n and pred[hi] == pm.LABELS[t]:
                    hi += 1
                over = (a - lo) + (hi - b)
                under = (b - a) - (e0 - s0)
                k = "one_piece_overruns" if over >= under else "one_piece_falls_short"
        why[k] = why.get(k, 0) + 1
    out["unmatched_at_0.5"] = why
    out["n_matched_0.5"] = int((best >= 0.5).sum())

    # -- timing ---------------------------------------------------------------------------------
    tol = 5 * FPS
    idx = np.arange(1, n)
    live_pair = L[1:] & L[:-1]
    for d, (x, y) in {"H>A": ("H", "A"), "A>H": ("A", "H")}.items():
        true_at = idx[live_pair & (g[:-1] == x) & (g[1:] == y)]
        pred_at = idx[(team[:-1] == x) & (team[1:] == y) & graded[1:] & graded[:-1]]
        off, spare = nearest_offsets(true_at, pred_at, tol)
        o = out.setdefault("turnover_offsets", [])
        o += off.tolist()
        out["false_flips"] = out.get("false_flips", 0) + spare
        out["pred_flips"] = out.get("pred_flips", 0) + len(pred_at)
    stops_t, stops_p = idx[L[:-1] & (g[1:] == "D")], idx[M[:-1] & (pred[1:] == "D")]
    starts_t, starts_p = idx[(g[:-1] == "D") & L[1:]], idx[(pred[:-1] == "D") & M[1:]]
    off, spare = nearest_offsets(stops_t, stops_p, tol)
    out["stop_offsets"], out["false_stops"], out["pred_stops"] = off.tolist(), spare, len(stops_p)
    off, spare = nearest_offsets(starts_t, starts_p, tol)
    out["restart_offsets"], out["false_restarts"] = off.tolist(), spare

    # the whole prediction slid earlier (negative) or later in time
    sh = {}
    for k in SHIFTS:
        p2 = np.roll(pred, k)
        p2[g == "X"] = "X"
        sh[str(k)] = S(np.where(p2 == "X", pred, p2))
    out["shift"] = sh
    # team call alone slid, stoppages kept
    sh = {}
    for k in SHIFTS:
        sh[str(k)] = S(lib.compose(M, np.roll(team, k), g))
    out["shift_team_only"] = sh
    return out


def main():
    tag = sys.argv[1]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, [(tag, n) for n in lib.match_names()]))
    path = os.path.join(lib.ROOT, "data", "cache", "diag", tag + ".json")
    json.dump(rows, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    main()
