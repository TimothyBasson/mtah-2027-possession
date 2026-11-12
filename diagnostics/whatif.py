"""Cheap what-ifs on saved predictions: paired noise between two versions, and stoppages redrawn
earlier or later.   python diagnostics/whatif.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

PROV = ["idsse", "metrica", "skillcorner"]


def paired(a, b, label):
    """Per-match difference b - a: provider means, sd between matches, and the standard error of the
    three-provider mean of the difference."""
    names = lib.match_names()
    d = {p: np.array([b[n] - a[n] for n in names if lib.provider(n) == p]) for p in PROV}
    mean = np.mean([d[p].mean() for p in PROV])
    se = np.sqrt(sum(d[p].var(ddof=1) / len(d[p]) for p in PROV)) / 3
    print(f"{label:34s} " + "  ".join(f"{p[:5]} {d[p].mean():+.3f} (sd {d[p].std(ddof=1):.3f}, up {int((d[p] > 0).sum())}/{len(d[p])})" for p in PROV)
          + f"   mean {mean:+.4f} se {se:.4f}")


def scores(tag, key="pred_heldout"):
    return {n: lib.spell_f1(lib.truth(n), lib.preds(tag, n)[key]) for n in lib.match_names()}


def redraw(pred, team, early, late):
    """Every predicted stoppage started `early` frames sooner and ended `late` frames later (negative
    shrinks it), the freed or covered frames taking the model's team call."""
    dead = pred == "D"
    out = dead.copy()
    for a, b in lib.runs_of(dead):
        out[a:b] = False
        out[max(0, a - early):max(0, b + late)] = True
    p = np.where(out, "D", team)
    p[pred == "X"] = "X"
    return p


if __name__ == "__main__":
    have = [t for t in ("main", "team", "team_mt15", "team_mt10") if os.path.isdir(os.path.join(lib.ROOT, "data", "cache", "diag", t))]
    s = {t: scores(t) for t in have}
    s["main_in"], s["team_in"] = scores("main", "pred_insample"), scores("team", "pred_insample")
    print("-- paired differences in spellF1, per match --")
    paired(s["main"], s["team"], "spread lean -> team model")
    paired(s["main"], s["main_in"], "main: held-out -> in-sample fit")
    paired(s["team"], s["team_in"], "team: held-out -> in-sample fit")
    for t in have[2:]:
        paired(s["team"], s[t], f"team -> {t}")
    for tag in ("main", "team"):
        print(f"-- {tag}: predicted stoppages redrawn (start earlier by / end later by, seconds): held-out mean spellF1 --")
        names = lib.match_names()
        data = [(n, lib.truth(n), lib.preds(tag, n)) for n in names]
        for early in (0, 12, 25, 37, 50, 62):
            row = []
            for late in (-25, -12, 0, 12, 25, 37):
                sc = {n: lib.spell_f1(g, redraw(z["pred_heldout"], z["team"], early, late)) for n, g, z in data}
                per = [np.mean([sc[n] for n in names if lib.provider(n) == p]) for p in PROV]
                row.append(f"{np.mean(per):.3f} ({per[0]:.3f}/{per[1]:.3f}/{per[2]:.3f})")
            print(f"start {early / 25:+.1f}s earlier | " + " | ".join(row))
        print("columns: end " + ", ".join(f"{x / 25:+.1f}s" for x in (-25, -12, 0, 12, 25, 37)) + "   cell: mean (idsse/metrica/skillcorner)")
