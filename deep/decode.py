"""Per-frame probabilities -> labels -> held-out spellF1.

    python deep/decode.py --tag v2

For each held-out provider, reads the probabilities deep/train.py saved, picks the decode settings
on that fold's validation matches (training providers, not trained on), and scores the held-out
provider's matches. Mean over providers as external/heldout.py: each provider counts equally.

Decode: a three-state chain (H, A, D) with one switching cost, Viterbi per half, on log
probabilities raised to 25 fps; then short runs absorbed as possession/model.py does.
"""

import argparse
import glob
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import data as D  # noqa: E402

LETTERS = np.array(list("HAD"))


def upsample(p, n):
    """(T, 3) at every STEP-th frame -> (n, 3) by linear interpolation."""
    t = np.arange(len(p)) * D.STEP
    return np.stack([np.interp(np.arange(n), t, p[:, k]) for k in range(p.shape[1])], 1)


def viterbi(logp, cost):
    """Best path through (n, S) log-probabilities with a flat cost for any change of state."""
    n, S = logp.shape
    back = np.zeros((n, S), np.int8)
    v = logp[0].copy()
    for i in range(1, n):
        best = v.argmax()
        stay = v >= v[best] - cost
        back[i] = np.where(stay, np.arange(S), best)
        v = np.where(stay, v, v[best] - cost) + logp[i]
    path = np.zeros(n, np.int8)
    path[-1] = v.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def absorb(codes, min_len):
    """Runs shorter than min_len[state] frames join the longer neighbour, shortest first."""
    runs = [[k, len(list(g))] for k, g in itertools.groupby(codes)]
    while len(runs) > 1:
        short = [i for i, (k, n) in enumerate(runs) if n < min_len[k]]
        if not short:
            break
        i = min(short, key=lambda i: runs[i][1])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: runs[j][1])
        runs[j][1] += runs.pop(i)[1]
        k = 1
        while k < len(runs):
            if runs[k - 1][0] == runs[k][0]:
                runs[k - 1][1] += runs.pop(k)[1]
            else:
                k += 1
    return np.array([k for k, n in runs for _ in range(n)], np.int8)


def labels_from(probs, spans, n, cost, min_team, min_dead, power=1.0):
    out = np.full(n, "X", "<U1")
    for p, (a, b) in zip(probs, spans):
        lp = power * np.log(np.clip(upsample(p, b - a + 1), 1e-6, 1))
        path = viterbi(lp, cost) if cost > 0 else lp.argmax(1).astype(np.int8)
        if min_team or min_dead:
            path = absorb(path, {0: min_team * 25, 1: min_team * 25, 2: min_dead * 25})
        out[a:b + 1] = LETTERS[path]
    return out


def score_file(args):
    path, combos = args
    z = np.load(path)
    name = os.path.basename(path)[:-4]
    d = np.load(D.match_path(name))
    spans = [tuple(h) for h in d["halves"]]
    truth = pm.label_codes(d["labels"])
    probs = [z[f"p{i}"] for i in range(len(spans))]
    rows = []
    for c in combos:
        r = pm.evaluate(truth, pm.label_codes(labels_from(probs, spans, len(truth), **c)), bootstrap_n=0)
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]))
    return name, str(z["kind"]), rows


GRID = [dict(cost=c, min_team=t, min_dead=dd) for c in (0, 2, 4, 8, 16, 32) for t, dd in ((0, 0), (1, 1), (2, 3))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    base = os.path.join(ROOT, "data", "cache", "deep", a.tag)
    means = []
    print(f"{'held out':12s}  spellF1  macroF1  alive/dead  team   best here   picked on validation")
    with ProcessPoolExecutor() as ex:
        for fold in sorted(os.listdir(base)):
            files = sorted(glob.glob(os.path.join(base, fold, "*.npz")))
            res = list(ex.map(score_file, [(f, GRID) for f in files]))
            val = [np.array(r) for n, k, r in res if k == "val"]
            test = [np.array(r) for n, k, r in res if k == "test"]
            if not test:
                continue
            # validation: each training provider counts equally
            provs = sorted({n.split("_")[0] for n, k, r in res if k == "val"})
            v = np.mean([np.mean([np.array(r) for n, k, r in res if k == "val" and n.startswith(p)], 0)
                         for p in provs], 0)
            pick = int(v[:, 0].argmax())
            t = np.mean(test, 0)
            means.append(t[pick])
            print(f"{fold:12s}  " + "  ".join(f"{x:7.3f}" for x in t[pick]) + f"   {t[:, 0].max():.3f}   {GRID[pick]}")
    print(f"{'mean':12s}  " + "  ".join(f"{x:7.3f}" for x in np.mean(means, 0)))


if __name__ == "__main__":
    main()
