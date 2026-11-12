"""Shared pieces for the spellF1 diagnostics: loading a match with a model's saved predictions,
scoring, and the 'what if this kind of mistake were fixed' rebuilds."""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
IOUS = pm.SPELL_IOUS


def provider(name):
    return os.path.basename(name).split("_")[0]


def match_names():
    return [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))]


def truth(name):
    return np.load(os.path.join(ROOT, "data", "external", name + ".npz"))["labels"]


def preds(tag, name):
    return np.load(os.path.join(ROOT, "data", "cache", "diag", tag, name + ".npz"))


def spell_f1(g, p, per_iou=False):
    """spellF1 of string labels p against g, exactly as the scorer computes it."""
    gc, pc = pm.label_codes(g), pm.label_codes(p)
    runs = pm.valid_runs(gc)
    s = pm.spell_scores(gc, pm.fill_abstentions(pc, runs), runs)
    return (s["mean_f1"], s) if per_iou else s["mean_f1"]


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def spells(labels):
    """[(start, stop, team)] maximal H or A runs; an X or a D ends one, as in the scorer."""
    out = []
    for t in "HA":
        out += [(a, b, t) for a, b in runs_of(labels == t)]
    return sorted(out)


def matching(g, p):
    """Per truth spell and per predicted spell: the share of the five IoU thresholds it is matched
    at (0..1), using the scorer's own greedy matching. Also the spell lists."""
    gc, pc = pm.label_codes(g), pm.label_codes(p)
    runs = pm.valid_runs(gc)
    pc = pm.fill_abstentions(pc, runs)
    gs, ps = pm.possession_spells(gc, runs), pm.possession_spells(pc, runs)
    ov = pm.spell_overlaps(gs, ps)
    gm, pmt = np.zeros(len(gs)), np.zeros(len(ps))
    for th in IOUS:
        gflag, pflag = np.zeros(len(gs), bool), np.zeros(len(ps), bool)
        for iou, i, j in ov:
            if iou < th:
                break
            if gflag[i] or pflag[j]:
                continue
            gflag[i] = pflag[j] = True
        gm += gflag
        pmt += pflag
    best = np.zeros(len(gs))
    for iou, i, j in ov:
        best[i] = max(best[i], iou)
    return gs, ps, gm / len(IOUS), pmt / len(IOUS), best


def fill_team(g):
    """The truth's team on every graded frame: its H/A where live, carried forward through
    stoppages (back-filled at the head of a run)."""
    gc = pm.label_codes(g)
    runs = pm.valid_runs(gc)
    t = np.where(np.isin(g, ["H", "A"]), gc, pm.X).astype(gc.dtype)
    filled = pm.fill_abstentions(t, runs)
    out = np.where(filled == pm.H, "H", "A")
    return out


def compose(alive, team, g):
    """Labels from an alive mask and a team call; X wherever the truth is X."""
    out = np.where(alive, team, "D")
    out[g == "X"] = "X"
    return out


# -- classes of mistake -----------------------------------------------------------------------
# Each returns {class name: boolean frame mask of the frames that class got wrong}.

def team_error_classes(g, team):
    """Wrong-team runs inside each true possession, by where they sit in it:
    whole (the possession is never given to its team), late_start (the model switches to the team
    late), early_end (it switches away early), interior (a false excursion to the other team in
    the middle). late/early are split by what is on the other side of that edge in the truth:
    a turnover (the other team) or a stoppage / the edge of the graded stretch."""
    names = ["whole_under2s", "whole_2to5s", "whole_over5s", "late_start_turnover", "late_start_restart",
             "early_end_turnover", "early_end_stoppage", "interior"]
    out = {k: np.zeros(len(g), bool) for k in names}
    n = len(g)
    for a, b, t in spells(g):
        wrong = team[a:b] != t
        if not wrong.any():
            continue
        for s, e in runs_of(wrong):
            if s == 0 and e == b - a:
                d = (b - a) / FPS
                k = "whole_under2s" if d < 2 else "whole_2to5s" if d < 5 else "whole_over5s"
            elif s == 0:
                k = "late_start_turnover" if a > 0 and g[a - 1] in "HA" else "late_start_restart"
            elif e == b - a:
                k = "early_end_turnover" if b < n and g[b] in "HA" else "early_end_stoppage"
            else:
                k = "interior"
            out[k][a + s:a + e] = True
    return out


def dead_error_classes(g, alive):
    """Alive/dead mistakes by where they sit:
    false_dead_whole     a whole true possession called dead
    false_dead_edge      dead called over the start or end of a possession (a stoppage drawn too long,
                         or a turnover called a stoppage)
    false_dead_interior  a stoppage invented in the middle of a possession (splits it)
    missed_dead_whole    a whole true stoppage called alive (its two possessions run together)
    missed_dead_edge     the start or end of a true stoppage called alive (stoppage drawn too short)"""
    names = ["false_dead_whole", "false_dead_edge", "false_dead_interior", "missed_dead_whole",
             "missed_dead_edge", "missed_dead_interior"]
    out = {k: np.zeros(len(g), bool) for k in names}
    for a, b, _ in spells(g):
        for s, e in runs_of(~alive[a:b]):
            k = "whole" if (s == 0 and e == b - a) else "edge" if (s == 0 or e == b - a) else "interior"
            out["false_dead_" + k][a + s:a + e] = True
    for a, b in runs_of(g == "D"):
        for s, e in runs_of(alive[a:b]):
            k = "whole" if (s == 0 and e == b - a) else "edge" if (s == 0 or e == b - a) else "interior"
            out["missed_dead_" + k][a + s:a + e] = True
    return out
