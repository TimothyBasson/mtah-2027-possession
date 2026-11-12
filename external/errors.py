"""Where the model's mistakes are, on the outside matches, held out.

    uv run python external/errors.py [--model DIR] [param=value ...]

Each provider is predicted with weights fitted on the other two (as external/heldout.py does),
then every graded frame the model gets wrong is put in one bucket:

  team, near a turnover    alive in both, wrong team, within NEAR s of a true change of team
  team, mid-possession     alive in both, wrong team, further from one
  false stoppage           truth in play, model says dead (near / away from a true stoppage edge)
  missed stoppage          truth dead, model says in play: near its start, near its end, or the middle

and every true possession the model fails to match at IoU 0.5 (spellF1's headline) by why:

  merged     the best same-team prediction overlapping it is over twice as long
  split      it is under half as long
  shifted    about the right length, in the wrong place
  wrong team no prediction for that team overlaps it

Plus how late the model calls each true change of team, and how short true possessions are,
which says what the 2 s minimum possession can and can't see. Providers weigh equally.
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from heldout import fit_fold, provider  # noqa: E402
from score import drop_ball, frames, pm  # noqa: E402

NEAR = 2.0  # s; a turnover or a stoppage edge this close counts as a timing error


def runs_of(x):
    """[(start, stop, value)] maximal runs, stop exclusive."""
    cut = np.flatnonzero(x[1:] != x[:-1]) + 1
    s, e = np.r_[0, cut], np.r_[cut, len(x)]
    return [(a, b, x[a]) for a, b in zip(s, e)]


def dist_to(idx, n):
    """Frames from each frame to the nearest index in idx (inf if none)."""
    if not len(idx):
        return np.full(n, np.inf)
    i = np.arange(n)
    j = np.clip(np.searchsorted(idx, i), 1, len(idx)) - 1
    return np.minimum(np.abs(i - idx[j]), np.abs(i - idx[np.minimum(j + 1, len(idx) - 1)]))


def spells(lab, halves):
    """H/A runs within each half, stoppages excluded: [(start, stop, team)]."""
    out = []
    for a, b in halves:
        out += [(a + s, a + e, k) for s, e, k in runs_of(lab[a:b + 1]) if k in "HA"]
    return out


def analyse(args):
    path, model_dir, params = args
    sys.path.insert(0, model_dir)
    import features
    import model
    d = np.load(path)
    halves = [tuple(h) for h in d["halves"]]
    mask = np.zeros(len(d["labels"]), bool)
    for s, e in halves:
        mask[s:e + 1] = True
    features.PITCH = model.PITCH = tuple(d["pitch"])
    model.HALVES = halves
    model.graded_mask = lambda n: mask
    F = frames(d["players"], drop_ball(d["ball"], 1.0, model.FPS, labels=d["labels"]), d["team"], features)
    pred, truth = model.predict(F, params), d["labels"]
    fps, n = model.FPS, len(truth)
    near = NEAR * fps
    g = mask & np.isin(truth, ["H", "A", "D"])

    # true events: changes of team (between consecutive possessions, through a stoppage or not)
    ts = spells(truth, halves)
    turn = np.array([s for (_, _, k0), (s, _, k) in zip(ts, ts[1:]) if k != k0], int)
    dr = [(h0 + a, h0 + b) for h0, h1 in halves for a, b, k in runs_of(truth[h0:h1 + 1]) if k == "D"]
    d_edges = np.sort(np.r_[[a for a, _ in dr], [b for _, b in dr]]).astype(int)
    to_turn, to_edge = dist_to(turn, n), dist_to(d_edges, n)
    to_start, to_end = dist_to(np.array([a for a, _ in dr], int), n), dist_to(np.array([b for _, b in dr], int), n)

    ta, pa = np.isin(truth, ["H", "A"]), np.isin(pred, ["H", "A"])
    wrong_team = g & ta & pa & (truth != pred)
    false_d = g & ta & (pred == "D")
    missed_d = g & (truth == "D") & pa
    b = {
        "team, near a turnover": wrong_team & (to_turn <= near),
        "team, mid-possession": wrong_team & (to_turn > near),
        "false stoppage, near a true edge": false_d & (to_edge <= near),
        "false stoppage, in open play": false_d & (to_edge > near),
        "missed stoppage, its start": missed_d & (to_start <= near) & (to_start <= to_end),
        "missed stoppage, its end": missed_d & (to_end <= near) & (to_end < to_start),
        "missed stoppage, its middle": missed_d & (np.minimum(to_start, to_end) > near),
    }
    frac = {k: v.sum() / g.sum() for k, v in b.items()}

    # possessions: why a true one goes unmatched at IoU 0.5
    ps = spells(pred, halves)
    why = dict(matched=0, merged=0, split=0, shifted=0, **{"wrong team": 0})
    for a, e, k in ts:
        best, blen = 0.0, 0
        for pa_, pe, pk in ps:
            if pk != k or pe <= a or pa_ >= e:
                continue
            iou = (min(e, pe) - max(a, pa_)) / (max(e, pe) - min(a, pa_))
            if iou > best:
                best, blen = iou, pe - pa_
        if best >= 0.5:
            why["matched"] += 1
        elif best == 0:
            why["wrong team"] += 1
        elif blen > 2 * (e - a):
            why["merged"] += 1
        elif blen < (e - a) / 2:
            why["split"] += 1
        else:
            why["shifted"] += 1

    # lateness: signed s from each true change of team to the nearest predicted one the same way
    pt = [(s, k) for (_, _, k0), (s, _, k) in zip(ps, ps[1:]) if k != k0]
    lag = []
    for s in turn:
        k = truth[s]
        c = [p - s for p, pk in pt if pk == k and abs(p - s) <= 10 * fps]
        lag.append(min(c, key=abs) / fps if c else np.nan)
    tl = np.array([(e - a) / fps for a, e, _ in ts])
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)
    return dict(frac=frac, wrong=(g & (truth != pred)).sum() / g.sum(), why=why, lag=np.array(lag),
                n_true=len(ts), n_pred=len(ps), true_len=tl, spellF1=r["spell_f1_mean"],
                turn_per_spell=len(turn) / max(len(ts), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "possession"))
    ap.add_argument("params", nargs="*")
    a = ap.parse_args()
    params = {k: (v if v.isalpha() else float(v)) for k, v in (p.split("=") for p in a.params)}
    model_dir = os.path.abspath(a.model)
    paths = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "data", "external", "*.npz")))
    sets = sorted({provider(p) for p in paths})
    with ProcessPoolExecutor() as ex:
        fits = dict(zip(sets, ex.map(fit_fold, [(model_dir, [p for p in paths if provider(p) != s]) for s in sets])))
        res = list(ex.map(analyse, [(p, model_dir, params | fits[provider(p)]) for p in paths]))
    by = {s: [r for p, r in zip(paths, res) if provider(p) == s] for s in sets}

    def eq(f):  # provider-equal mean of a per-match number
        return np.mean([np.mean([f(r) for r in by[s]]) for s in sets]), [np.mean([f(r) for r in by[s]]) for s in sets]

    print(f"held out, providers equal; columns: mean, {', '.join(sets)}\n")
    m, per = eq(lambda r: r["wrong"])
    print(f"{'graded frames wrong':36s} {m:6.1%}   " + "  ".join(f"{x:6.1%}" for x in per))
    for k in res[0]["frac"]:
        m, per = eq(lambda r: r["frac"][k])
        print(f"  {k:34s} {m:6.1%}   " + "  ".join(f"{x:6.1%}" for x in per))
    print("\ntrue possessions (IoU 0.5):")
    for k in res[0]["why"]:
        m, per = eq(lambda r: r["why"][k] / r["n_true"])
        print(f"  {k:34s} {m:6.1%}   " + "  ".join(f"{x:6.1%}" for x in per))
    m, per = eq(lambda r: r["n_pred"] / r["n_true"])
    print(f"{'predicted / true possessions':36s} {m:6.2f}   " + "  ".join(f"{x:6.2f}" for x in per))
    m, per = eq(lambda r: r["turn_per_spell"])
    print(f"{'true possessions ending in a turnover':36s} {m:6.1%}   " + "  ".join(f"{x:6.1%}" for x in per))
    print("\ntrue changes of team: model's switch, s after (- before)")
    for s in sets:
        lag = np.concatenate([r["lag"] for r in by[s]])
        f = np.isfinite(lag)
        print(f"  {s:12s} found within 10 s {f.mean():5.1%}   median {np.median(lag[f]):+5.2f}   "
              f"25-75% {np.percentile(lag[f], 25):+5.2f} {np.percentile(lag[f], 75):+5.2f}")
    print("\ntrue possession length, s (10 / 25 / 50 / 75 / 90%), share under 2 s and under 4 s")
    for s in sets:
        tl = np.concatenate([r["true_len"] for r in by[s]])
        print(f"  {s:12s} " + " ".join(f"{x:5.1f}" for x in np.percentile(tl, [10, 25, 50, 75, 90]))
              + f"   {(tl < 2).mean():5.1%} {(tl < 4).mean():5.1%}")


if __name__ == "__main__":
    main()
