"""Shared pieces of the touch work: truths, saved held-out probabilities, a decode that takes
per-frame team evidence, the merge slice, and scoring by provider."""

import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402
from turnover import common as TC  # noqa: E402

PROVIDERS = ("idsse", "metrica", "skillcorner")
RUNS = os.path.join(ROOT, "data", "cache", "alldata", "runs")
LAB = os.path.join(ROOT, "data", "cache", "alldata", "labels")
TOUCH = os.path.join(ROOT, "data", "cache", "alldata", "touch")
H, A, DEAD, X = (int(v) for v in pm.label_codes(np.array(list("HADX"))))
RAW = dict(cost=4, min_team=1, min_dead=1)
FOLDED = dict(cost=4, min_team=2, min_dead=3)
SLICE_MAX = 3.0  # s: a "brief touch" possession


def match_path(name):
    return os.path.join(ROOT, "data", "external", name + ".npz")


def truths(name):
    f = os.path.join(LAB, name + ".npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["raw"], z["fold"], [tuple(int(v) for v in h) for h in z["halves"]]
    from control import control_labels
    d = dict(np.load(match_path(name)))
    raw, fold = pm.label_codes(d["labels"]), pm.label_codes(control_labels(d))
    os.makedirs(LAB, exist_ok=True)
    np.savez_compressed(f, raw=raw, fold=fold, halves=d["halves"])
    return raw, fold, [tuple(int(v) for v in h) for h in d["halves"]]


def test_matches(tag):
    """[(fold, name)] of the held-out matches of a saved run."""
    out = []
    for fold in PROVIDERS:
        for f in sorted(glob.glob(os.path.join(RUNS, tag, fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                out.append((fold, os.path.basename(f)[:-4]))
    return out


def probs(tag, name, key="s", pre=""):
    z = np.load(os.path.join(RUNS, tag, name.split("_")[0], name + ".npz"))
    n = sum(1 for k in z.files if k.startswith(pre + key) and k[len(pre + key):].isdigit())
    return [z[f"{pre}{key}{i}"] for i in range(n)]


def decode_ev(probs, spans, n, ev=None, cost=4, min_team=1, min_dead=1, keep=None):
    """Codes (pm's) per frame. ev: (n, 3) added to the log-probabilities of H, A, D before the
    chain. keep: (n,) bool, frames of forced possessions; a run that holds one is never absorbed
    for being short."""
    out = np.full(n, X, np.int64)
    for p, (a, b) in zip(probs, spans):
        lp = np.log(np.clip(decode.upsample(p, b - a + 1), 1e-6, 1))
        if ev is not None:
            lp = lp + ev[a:b + 1]
        path = TC.viterbi_ev(lp, cost)
        if min_team or min_dead:
            if keep is None:
                path = TC.absorb(path, min_team, min_dead)
            else:
                path = absorb_keep(path, keep[a:b + 1], min_team, min_dead)
        out[a:b + 1] = np.array([H, A, DEAD])[path]
    return out


def absorb_keep(path, keep, min_team, min_dead):
    """TC.absorb, but runs that contain a kept frame survive: they are given a fourth and fifth
    state code for the absorb (min length 0) and mapped back."""
    e = np.flatnonzero(np.diff(path) != 0) + 1
    s, t = np.r_[0, e], np.r_[e, len(path)]
    tmp = path.astype(np.int8).copy()
    ck = np.r_[0, np.cumsum(keep)]
    for a, b in zip(s, t):
        if path[a] < 2 and ck[b] - ck[a] > 0:
            tmp[a:b] = 3 + path[a]
    out = TC._absorb(tmp, np.array([min_team * 25, min_team * 25, min_dead * 25, 0, 0], np.int64))
    return np.where(out >= 3, out - 3, out).astype(np.int8)


def spell(t, p):
    return pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"]


def merge_slice(t, p, max_len=SLICE_MAX):
    """The knock-on slice of one match: true possessions under max_len s that sit directly between
    two possessions of the other team (the brief touch), and those two neighbours. Returns counts
    [touches, touches found, neighbours, neighbours found, triples, triples merged, triples all
    found]; found = matched at IoU 0.5; merged = one predicted possession of the neighbours' team
    covers over half of each of the three."""
    runs = pm.valid_runs(t)
    gts, prs = pm.possession_spells(t, runs), pm.possession_spells(p, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gts, prs), len(gts), 0.5)
    c = np.zeros(7)
    nb = set()
    for i in range(1, len(gts) - 1):
        a, b, k = gts[i]
        pa, pb, pk = gts[i - 1]
        na, nb_, nk = gts[i + 1]
        if (b - a) >= max_len * 25 or pb != a or na != b or pk == k or nk == k:
            continue
        c[0] += 1; c[1] += flags[i]
        nb.update((i - 1, i + 1))
        c[4] += 1
        c[6] += flags[i - 1] and flags[i] and flags[i + 1]
        seg = p[pa:nb_]
        merged = (p[a:b] == pk).mean() > 0.5 and (p[pa:pb] == pk).mean() > 0.5 and (p[na:nb_] == pk).mean() > 0.5
        c[5] += merged
    c[2] = len(nb); c[3] = sum(flags[i] for i in nb)
    return c


def evaluate(name, p_raw, p_fold):
    raw, fold, _ = truths(name)
    return dict(raw=spell(raw, p_raw), fold=spell(fold, p_fold), sl=merge_slice(raw, p_raw),
                n=(len(pm.possession_spells(raw, pm.valid_runs(raw))), len(pm.possession_spells(p_raw, pm.valid_runs(raw)))))


def report(title, res, base=None):
    """res: {name: evaluate()}. One line: as built / folded per provider and mean, then the slice."""
    names = sorted(res)
    per = {k: [np.mean([res[n][k] for n in names if n.startswith(pv)]) for pv in PROVIDERS] for k in ("raw", "fold")}
    sl = np.sum([res[n]["sl"] for n in names], 0)
    slp = [np.sum([res[n]["sl"] for n in names if n.startswith(pv)], 0) for pv in PROVIDERS]
    line = f"{title:34s} built " + " ".join(f"{x:.3f}" for x in per["raw"]) + f" | {np.mean(per['raw']):.3f}"
    line += "  folded " + " ".join(f"{x:.3f}" for x in per["fold"]) + f" | {np.mean(per['fold']):.3f}"
    line += "  touch found " + " ".join(f"{s[1] / max(s[0], 1):.2f}" for s in slp)
    line += "  nb found " + " ".join(f"{s[3] / max(s[2], 1):.2f}" for s in slp)
    line += "  merged " + " ".join(f"{s[5] / max(s[4], 1):.2f}" for s in slp)
    if base is not None:
        for k in ("raw", "fold"):
            d = np.array([res[n][k] - base[n][k] for n in names])
            line += f"  d{k} {d.mean():+.3f} up {(d > 0).sum()}/{len(d)}"
    print(line, flush=True)
    return per, sl


def pool_map(fn, jobs, workers=4):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs))


def player_dists(d, ball_xy):
    """(n,) nearest home and away player distance to the given ball (n, 2), NaN where none."""
    pl = d["players"]
    dist = np.hypot(pl[:, :, 0] - ball_xy[:, None, 0], pl[:, :, 1] - ball_xy[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(dist[:, d["team"] == 0], axis=1), np.nanmin(dist[:, d["team"] == 1], axis=1)
