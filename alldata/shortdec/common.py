"""Shared pieces for the short-possession decode work: saved held-out probabilities of the
all-data model, a fast spellF1 that equals the organisers' (checked in check()), and the
"merge slice": brief true possessions between two possessions of the other team, with those two.

Probabilities are the two heads averaged, as alldata/learned/run.py decodes them for the file.
"""
import glob
import os
import sys

import numba
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned")]
import score as S  # noqa: E402
from score import pm, decode  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "turnover"))
PROVS = ("idsse", "metrica", "skillcorner")
H, A, D, X = (int(v) for v in pm.label_codes(np.array(list("HADX"))))
assert (H, A, D) == (0, 1, 2)
IOUS = np.array(pm.SPELL_IOUS)
SHORT = 3.0  # s: a "brief" possession for the merge slice


def matches(tag, kind="test"):
    """[(fold, name, path)] of a saved run."""
    out = []
    for fold in PROVS:
        for f in sorted(glob.glob(os.path.join(S.tag_dir(tag), fold, "*.npz"))):
            if str(np.load(f)["kind"]) == kind:
                out.append((fold, os.path.basename(f)[:-4], f))
    return out


def load(path, key="s", head="both"):
    """Per half: (T, 3) probabilities at 5 fps; and the truths (as built, folded) and spans."""
    name = os.path.basename(path)[:-4]
    raw, fold, spans = S.truths(name)
    z = np.load(path)
    probs = []
    for i in range(len(spans)):
        a, b = z[f"{key}{i}"], z[f"f{key}{i}"]
        probs.append((a + b) / 2 if head == "both" else a if head == "built" else b)
    return probs, spans, raw.astype(np.int8), fold.astype(np.int8)


@numba.njit(cache=True)
def _spells(codes):
    """Maximal H/A runs of a code array with no X inside: (start, stop, code) rows."""
    n = len(codes)
    out = np.empty((n, 3), np.int64)
    r = 0
    i = 0
    while i < n:
        j = i
        while j < n and codes[j] == codes[i]:
            j += 1
        if codes[i] < 2:
            out[r, 0] = i; out[r, 1] = j; out[r, 2] = codes[i]; r += 1
        i = j
    return out[:r]


def spells(codes, runs):
    out = []
    for a, b in runs:
        s = _spells(codes[a:b]); s[:, :2] += a
        out.append(s)
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


@numba.njit(cache=True)
def _match(g, p, ious):
    """Greedy one-to-one matching as possession_metrics.match_spells: flags (len(ious), n_gt)."""
    cap = 4 * (len(g) + len(p)) + 16
    io = np.empty(cap); gi = np.empty(cap, np.int64); pj = np.empty(cap, np.int64)
    m = 0
    j0 = 0
    for i in range(len(g)):
        while j0 < len(p) and p[j0, 1] <= g[i, 0]:
            j0 += 1
        j = j0
        while j < len(p) and p[j, 0] < g[i, 1]:
            if p[j, 2] == g[i, 2]:
                inter = min(g[i, 1], p[j, 1]) - max(g[i, 0], p[j, 0])
                union = max(g[i, 1], p[j, 1]) - min(g[i, 0], p[j, 0])
                if m == cap:
                    return np.zeros((0, 0), np.bool_)
                io[m] = inter / union; gi[m] = i; pj[m] = j; m += 1
            j += 1
    # sort by (-iou, gt, pred): pairs were generated in (gt, pred) order, so a stable sort on -iou
    order = np.argsort(-io[:m], kind="mergesort")
    flags = np.zeros((len(ious), len(g)), np.bool_)
    for t in range(len(ious)):
        used = np.zeros(len(p), np.bool_)
        for o in order:
            if io[o] < ious[t]:
                break
            if flags[t, gi[o]] or used[pj[o]]:
                continue
            flags[t, gi[o]] = True; used[pj[o]] = True
    return flags


def spell_f1(truth, pred, runs=None, detail=False):
    """spellF1 (mean of F1 over the IoU thresholds), as possession_metrics.spell_scores."""
    runs = pm.valid_runs(truth) if runs is None else runs
    g, p = spells(truth, runs), spells(pred, runs)
    fl = _match(g, p, IOUS)
    tp = fl.sum(1)
    f1 = np.where(tp > 0, 2 * tp / np.maximum(len(g) + len(p), 1), 0.0)
    return (float(f1.mean()), g, p, fl) if detail else float(f1.mean())


def sandwich(g, short=SHORT):
    """Indices of brief true possessions that touch a possession of the other team on both sides."""
    n = len(g)
    i = np.arange(1, n - 1)
    ok = ((g[i, 1] - g[i, 0]) < short * 25) & (g[i - 1, 1] == g[i, 0]) & (g[i + 1, 0] == g[i, 1]) & \
         (g[i - 1, 2] != g[i, 2]) & (g[i + 1, 2] != g[i, 2])
    return i[ok]


def slice_stats(truth, pred, runs=None):
    """spellF1 and the merge slice: [n brief, brief found, n neighbours, neighbours found,
    triples merged (one drawn possession covers most of both neighbours), n true, n drawn],
    'found' at IoU 0.5."""
    runs = pm.valid_runs(truth) if runs is None else runs
    f1, g, p, fl = spell_f1(truth, pred, runs, detail=True)
    f = fl[2]
    mid = sandwich(g)
    nb = np.unique(np.r_[mid - 1, mid + 1])
    nb = nb[~np.isin(nb, mid)]
    merged = 0
    for i in mid:
        a0, b0 = g[i - 1, :2]; a1, b1 = g[i + 1, :2]
        k = g[i - 1, 2]
        seg = pred[a0 + (b0 - a0) // 2:a1 + (b1 - a1) // 2 + 1]  # middle of one neighbour to the middle of the other
        merged += bool((seg == k).all())
    return f1, np.array([len(mid), f[mid].sum(), len(nb), f[nb].sum(), merged, len(g), len(p)])


def provider_mean(names, vals):
    vals = np.asarray(vals, float)
    per = [vals[[n.startswith(p) for n in names]].mean(0) for p in PROVS]
    return np.array(per + [np.mean(per, 0)])


def provider_sum(names, vals):
    vals = np.asarray(vals, float)
    return np.array([vals[[n.startswith(p) for n in names]].sum(0) for p in PROVS])


def check(tag="pw1e"):
    """The fast spellF1 equals the organisers' on the shipped decode of three matches."""
    for fold, name, f in matches(tag)[::12]:
        probs, spans, raw, fold_t = load(f)
        pred = pm.label_codes(decode.labels_from(probs, spans, len(raw), cost=4, min_team=1, min_dead=1)).astype(np.int8)
        for t in (raw, fold_t):
            ref = pm.evaluate(t, pred, bootstrap_n=0)["spell_f1_mean"]
            mine = spell_f1(t, pm.fill_abstentions(pred, pm.valid_runs(t)))
            assert abs(ref - mine) < 1e-12, (name, ref, mine)
            print(name, round(ref, 4), "ok")


if __name__ == "__main__":
    check()
