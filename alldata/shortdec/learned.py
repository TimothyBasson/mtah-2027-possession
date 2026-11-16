"""A learned scorer for candidate brief touches, in place of one threshold on the peak.

    python alldata/shortdec/learned.py --tag pw1e [--target built|folded]

Candidates are the bumps of the other team's probability (head taught on labels as built, peak
0.08 or more) inside a possession of the shipped decode. A gradient-boosted classifier scores
each from the shape of the bump in both heads; it is fitted on each fold's validation matches
(training providers, not trained on) against whether the truth has the other team at the peak,
and the score threshold is picked there too (leave-one-match-out scores). Scored on the held-out
provider.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import dec  # noqa: E402
from common import pm  # noqa: E402

LO = 0.08
THETAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2.0)


def candidates(path, Eb, Ef, P, lo=LO, edge=10, min_run=50, rho=0.5, min_len=10, max_len=75):
    """[(start, stop, team, features)] for one half."""
    out = []
    n = len(path)
    e = np.flatnonzero(path[1:] != path[:-1]) + 1
    for i, j in zip(np.r_[0, e], np.r_[e, n]):
        k = path[i]
        if k >= 2 or j - i < min_run:
            continue
        o = 1 - k
        s = Eb[i:j, o]
        ab = (s >= lo).astype(np.int8)
        ed = np.flatnonzero(np.diff(np.r_[0, ab, 0]))
        for a, b in zip(ed[::2], ed[1::2]):
            if a <= edge or b >= (j - i) - edge:
                continue
            pk = a + int(np.argmax(s[a:b]))
            l, h = pk, pk + 1
            while l - 1 >= 1 and s[l - 1] >= rho * s[pk] and pk - (l - 1) < max_len:
                l -= 1
            while h < (j - i) - 1 and s[h] >= rho * s[pk] and h - l < max_len:
                h += 1
            while h - l < min_len and (l > 1 or h < (j - i) - 1):
                if l > 1:
                    l -= 1
                if h - l < min_len and h < (j - i) - 1:
                    h += 1
            w0, w1 = max(i, i + a - 75), min(j, i + b + 75)
            around = np.r_[Eb[w0:i + a, o], Eb[i + b:w1, o]]
            f = [s[pk], Ef[i + a:i + b, o].max(), P[i + a:i + b, o].max(), (h - l) / 25, (b - a) / 25,
                 P[i + a:i + b, k].min(), P[i + a:i + b, 2].max(), (j - i) / 25, min(pk, j - i - pk) / 25,
                 around.mean() if len(around) else 0.0, s[pk] - Ef[i + pk, o]]
            out.append((i + l, i + h, o, i + pk, f))
    return out


def one(args):
    fold, name, f, key = args
    probs, spans, raw, fd = C.load(f, key)
    hb = C.load(f, key, "built")[0]; hf = C.load(f, key, "folded")[0]
    base = np.full(len(raw), 3, np.int8)
    cands = []
    for p, pb, pf, (a, b) in zip(probs, hb, hf, spans):
        P, Eb, Ef = (dec.upsample(x, b - a + 1) for x in (p, pb, pf))
        path = dec.decode_half(P)
        base[a:b + 1] = path
        for l, h, o, pk, ft in candidates(path, Eb, Ef, P):
            cands.append((a + l, a + h, o, int(raw[a + pk] == o), int(fd[a + pk] == o), ft))
    return fold, name, base, cands, raw, fd


def score(base, cands, keep, raw, fd):
    pred = base.copy()
    for (l, h, o, *_), k in zip(cands, keep):
        if k:
            pred[l:h] = o
    out = []
    for t in (raw, fd):
        runs = pm.valid_runs(t)
        f1, st = C.slice_stats(t, pm.fill_abstentions(pred, runs), runs)
        out.append(np.r_[f1, st])
    return np.array(out)


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--key", default="s")
    ap.add_argument("--target", default="built", choices=["built", "folded"])
    a = ap.parse_args()
    ti = 3 if a.target == "built" else 4
    with ProcessPoolExecutor(4) as ex:
        test = list(ex.map(one, [(fo, n, f, a.key) for fo, n, f in C.matches(a.tag, "test")]))
        val = list(ex.map(one, [(fo, n, f, a.key) for fo, n, f in C.matches(a.tag, "val")]))
    names, rows, picked = [], {th: [] for th in THETAS}, []
    for fold in C.PROVS:
        V = [v for v in val if v[0] == fold]
        Xs = [np.array([c[5] for c in v[3]]) for v in V]
        ys = [np.array([c[ti] for c in v[3]]) for v in V]
        # threshold: leave-one-match-out scores on the validation matches, providers equal
        vs = {th: [] for th in THETAS}
        for m, v in enumerate(V):
            clf = fit(np.vstack([x for i, x in enumerate(Xs) if i != m]), np.concatenate([y for i, y in enumerate(ys) if i != m]))
            s = clf.predict_proba(Xs[m])[:, 1]
            for th in THETAS:
                vs[th].append(score(v[2], v[3], s >= th, v[4], v[5])[:, 0])
        provs = sorted({v[1].split("_")[0] for v in V})
        vm = {th: np.mean([np.mean([r for r, v in zip(vs[th], V) if v[1].startswith(p)], 0) for p in provs], 0) for th in THETAS}
        best = max(THETAS, key=lambda th: vm[th][0 if a.target == "built" else 1])
        clf = fit(np.vstack(Xs), np.concatenate(ys))
        prec = []
        for t in [t for t in test if t[0] == fold]:
            X = np.array([c[5] for c in t[3]])
            s = clf.predict_proba(X)[:, 1]
            y = np.array([c[ti] for c in t[3]])
            prec.append((y[s >= best].sum(), (s >= best).sum()))
            names.append(t[1])
            for th in THETAS:
                rows[th].append(score(t[2], t[3], s >= th, t[4], t[5]))
            picked.append(rows[best][-1])
        pr = np.sum(prec, 0)
        print(f"{fold}: threshold picked {best}; candidates fitted on {sum(len(y) for y in ys)}; held out it keeps {pr[1]} touches, {pr[0] / max(pr[1], 1):.0%} true at the peak")
    print("columns: IDSSE Metrica SkillCorner mean, labels as built || folded")
    for th in THETAS + ("picked",):
        R = np.array(picked if th == "picked" else rows[th])
        out = []
        for li in range(2):
            f = C.provider_mean(names, R[:, li, 0]); s = C.provider_sum(names, R[:, li, 1:]).sum(0)
            out.append(" ".join(f"{x:.3f}" for x in f) + f" | nb {s[3] / s[2]:.2f} merged {s[4] / s[0]:.2f}")
        print("  ||  ".join(out), "threshold", th if th != 2.0 else "none (shipped)")


if __name__ == "__main__":
    main()
