"""Explicit-duration (semi-Markov) decode: each possession or stoppage is one segment whose
length has its own prior, read off the training providers' labels, instead of one flat cost per
change. Works at 5 fps; a segment's score is w * 5 * sum(log p) + log prior(length | state)
+ log P(state | previous state).

    python alldata/shortdec/semimarkov.py --tag pw1e

For each held-out provider the priors come from the other two providers' labels (as built or
folded, both tried), and w and the prior are picked on the fold's validation matches.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import dec  # noqa: E402
from common import pm, S  # noqa: E402

DMAX = 300  # 60 s at 5 fps; longer segments pay the tail rate per extra frame


def priors(provs, which):
    """log P(length | state) at 5 fps frames (1..DMAX, then a geometric tail rate) and
    log P(next state | state), from the labels of the named providers."""
    cnt = np.ones((3, DMAX + 1)) * 0.05
    tr = np.ones((3, 3)) - np.eye(3)
    import glob
    for f in sorted(glob.glob(os.path.join(C.ROOT, "data", "external", "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name.split("_")[0] not in provs:
            continue
        raw, fold, spans = S.truths(name)
        t = raw if which == "built" else fold
        for a, b, k in pm.segments(t, pm.valid_runs(t)):
            d = max(1, int(round((b - a) / 5)))
            cnt[k, min(d, DMAX)] += 1
        for _, fr, to in pm.transitions(t, pm.valid_runs(t)):
            tr[fr, to] += 1
    logd = np.full((3, DMAX + 1), -1e9)
    tail = np.zeros(3)
    for k in range(3):
        c = cnt[k, 1:]
        # smooth the histogram over a few frames so single lengths do not matter
        c = np.convolve(np.r_[c[0], c[0], c, c[-1], c[-1]], np.ones(5) / 5, "valid")
        logd[k, 1:] = np.log(c / c.sum())
        tail[k] = -1.0 / max(np.average(np.arange(1, DMAX + 1), weights=c), 1)
    np.fill_diagonal(tr, 1e-12)  # a segment never follows one of its own state
    return logd, tail, np.log(tr / tr.sum(1, keepdims=True))


@numba.njit(cache=True)
def segment(lp, logd, tail, logtr):
    """Best segmentation of (T, 3) weighted log-probabilities."""
    T = len(lp)
    D = logd.shape[1] - 1
    cum = np.zeros((T + 1, 3))
    for t in range(T):
        cum[t + 1] = cum[t] + lp[t]
    best = np.full((T + 1, 3), -1e18)   # best score of a segmentation of [0, t) ending in state k
    bd = np.zeros((T + 1, 3), np.int64)
    bk = np.zeros((T + 1, 3), np.int64)
    # a segment longer than D is a D-long one extended frame by frame at the tail rate
    ext = np.full((T + 1, 3), -1e18)    # best score ending at t in state k with a segment of length >= D
    es = np.zeros((T + 1, 3), np.int64)  # ... and where it started
    for t in range(1, T + 1):
        for k in range(3):
            for d in range(1, min(D, t) + 1):
                s = t - d
                e = cum[t, k] - cum[s, k] + logd[k, d]
                if s == 0:
                    v, pk = e, -1
                else:
                    v, pk = -1e18, -1
                    for j in range(3):
                        if j != k and best[s, j] + logtr[j, k] > v:
                            v = best[s, j] + logtr[j, k]; pk = j
                    v += e
                if d == D and v > ext[t, k]:
                    ext[t, k] = v; es[t, k] = s
                if v > best[t, k]:
                    best[t, k] = v; bd[t, k] = d; bk[t, k] = pk
            if t > D:
                v = ext[t - 1, k] + lp[t - 1, k] + tail[k]
                if v > ext[t, k]:
                    ext[t, k] = v; es[t, k] = es[t - 1, k]
                if ext[t, k] > best[t, k]:
                    best[t, k] = ext[t, k]; bd[t, k] = t - es[t, k]; bk[t, k] = -2
    out = np.zeros(T, np.int8)
    t = T
    k = int(np.argmax(best[T]))
    while t > 0:
        d = bd[t, k]
        out[t - d:t] = k
        s = t - d
        pk = bk[t, k]
        if pk == -2:  # a long segment: its predecessor was chosen when it started
            v, pk = -1e18, -1
            for j in range(3):
                if j != k and s > 0 and best[s, j] + logtr[j, k] > v:
                    v = best[s, j] + logtr[j, k]; pk = j
        t = s
        k = pk if pk >= 0 else 0
    return out


COMBOS = [("shipped", None, None)] + [(wh, w, mn) for wh in ("built", "folded") for w in (0.1, 0.2, 0.4, 0.8) for mn in (0.0, 1.0)]


def one(args):
    fold, name, f, key = args
    probs, spans, raw, fd = C.load(f, key)
    others = [p for p in C.PROVS if p != fold]
    pri = {wh: priors(others, wh) for wh in ("built", "folded")}
    truths = [(t, pm.valid_runs(t)) for t in (raw, fd)]
    out = []
    for wh, w, mn in COMBOS:
        pred = np.full(len(raw), 3, np.int8)
        for p, (a, b) in zip(probs, spans):
            n = b - a + 1
            if wh == "shipped":
                pred[a:b + 1] = dec.decode_half(dec.upsample(p, n))
                continue
            path5 = segment(w * 5 * np.log(np.clip(p, 1e-6, 1)), *pri[wh])
            path = np.repeat(path5, 5)[:n]
            if len(path) < n:
                path = np.r_[path, np.full(n - len(path), path[-1], np.int8)]
            # a 5 fps frame stands for the 5 frames around it, as the upsampling does
            path = np.r_[path[2:], path[-2:]].astype(np.int8)
            if mn:
                path = dec.absorb(path, dec.upsample(p, n), int(mn * 25), int(mn * 25), int(mn * 25), 0.0)
            pred[a:b + 1] = path
        row = []
        for t, runs in truths:
            f1, st = C.slice_stats(t, pm.fill_abstentions(pred, runs), runs)
            row.append(np.r_[f1, st])
        out.append(row)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--key", default="s")
    a = ap.parse_args()
    res = {}
    for kind in ("test", "val"):
        ms = C.matches(a.tag, kind)
        with ProcessPoolExecutor(4) as ex:
            res[kind] = (ms, np.array(list(ex.map(one, [(fo, n, f, a.key) for fo, n, f in ms]))))
    ms, R = res["test"]
    names = [n for _, n, _ in ms]
    print("columns: IDSSE Metrica SkillCorner mean, labels as built || folded")
    for ci, c in enumerate(COMBOS):
        out = []
        for li in range(2):
            f = C.provider_mean(names, R[:, ci, li, 0]); s = C.provider_sum(names, R[:, ci, li, 1:]).sum(0)
            out.append(" ".join(f"{x:.3f}" for x in f) + f" | nb {s[3] / s[2]:.2f} merged {s[4] / s[0]:.2f} drawn/true {s[6] / s[5]:.2f}")
        print("  ||  ".join(out), c)
    msv, Rv = res["val"]
    for obj, w in (("as built", (1, 0)), ("folded", (0, 1))):
        per = np.zeros((2, 4)); picks = []
        for pi, fold in enumerate(C.PROVS):
            idx = [i for i, (fo, n, _) in enumerate(msv) if fo == fold]
            provs = sorted({msv[i][1].split("_")[0] for i in idx})
            v = np.mean([np.mean([Rv[i, :, :, 0] for i in idx if msv[i][1].startswith(p)], 0) for p in provs], 0)
            pk = int((v @ np.array(w)).argmax()); picks.append(COMBOS[pk])
            for li in range(2):
                per[li, pi] = np.mean([R[i, pk, li, 0] for i, n in enumerate(names) if n.startswith(fold)])
        per[:, 3] = per[:, :3].mean(1)
        print("picked on validation for", obj, ":", np.round(per[0], 3), "||", np.round(per[1], 3), picks)


if __name__ == "__main__":
    main()
