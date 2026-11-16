"""Detected touches as decode evidence: detector quality, then spellF1 and the merge slice.

    python alldata/touch/tuse.py --tag pw1e --touch t1 [--kind test] [--v 1]

Events: local peaks of the detector's touch probability above a threshold, team by the larger
side. In the decode a touch raises its team's log-probability by weight x score until the next
touch or `hold` seconds; with keep, a possession that starts at a touch is not absorbed for
being under the minimum length. --kind val scores the validation matches of each fold (trained
providers, matches never trained on), where the settings are picked.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

KEY = {1: "s", 4: "l"}


def events(q, thr, gap=6):
    """q (n, 2) touch probabilities -> frames, teams, scores of the peaks."""
    q = q.astype(np.float32)
    s = q.sum(1)
    k = np.ones(5) / 5
    sm = np.convolve(s, k, "same")
    mx = sm.copy()
    for d in range(1, gap + 1):
        mx[d:] = np.maximum(mx[d:], sm[:-d]); mx[:-d] = np.maximum(mx[:-d], sm[d:])
    f = np.flatnonzero((sm >= mx) & (sm > thr))
    if len(f) > 1:  # plateaus: keep the first of a run of equal neighbours
        f = f[np.r_[True, np.diff(f) > 1]]
    qs = np.stack([np.convolve(q[:, c], np.ones(7), "same") for c in (0, 1)], 1)
    return f, qs[f].argmax(1), sm[f]


def touch_probs(touch, fold, name, v, n, spans):
    z = np.load(os.path.join(C.TOUCH, "runs", touch, fold, name + ".npz"))
    q = np.zeros((n, 2), np.float32)
    for i, (a, b) in enumerate(spans):
        q[a:b + 1] = z[f"q{v}_{i}"]
    return q


def evidence(n, f, tm, sc, weight, hold):
    ev = np.zeros((n, 3))
    keep = np.zeros(n, bool)
    for j in range(len(f)):
        e = min(f[j] + int(hold * 25), f[j + 1] if j + 1 < len(f) else n)
        ev[f[j]:e, tm[j]] += weight * sc[j]
        keep[f[j]] = True
    return ev, keep


def quality(f, tm, tf, tt, tol=8):
    """Precision and recall of events against the recorded ball's touches (same team, +-tol frames)."""
    ok = tt >= 0
    tf, tt = tf[ok], tt[ok]
    hit_t = np.zeros(len(tf), bool)
    hit_e = np.zeros(len(f), bool)
    j0 = np.searchsorted(tf, f - tol)
    for i in range(len(f)):
        j = j0[i]
        while j < len(tf) and tf[j] <= f[i] + tol:
            if tt[j] == tm[i] and not hit_t[j]:
                hit_t[j] = hit_e[i] = True
                break
            j += 1
    return hit_e.sum(), len(f), hit_t.sum(), len(tf)


def one(args):
    fold, name, tag, touch, v, settings, thrs = args
    raw, fo, spans = C.truths(name)
    n = len(raw)
    key = KEY[v]
    z = np.load(os.path.join(C.RUNS, tag, fold, name + ".npz"))
    pr = [z[f"{key}{i}"] for i in range(len(spans))]
    pf = [z[f"f{key}{i}"] for i in range(len(spans))]
    q = touch_probs(touch, fold, name, v, n, spans)
    tz = np.load(os.path.join(C.TOUCH, "feats", name + ".npz"))
    out = {"base": C.evaluate(name, C.decode_ev(pr, spans, n, **C.RAW), C.decode_ev(pf, spans, n, **C.FOLDED))}
    for thr in thrs:
        f, tm, sc = events(q, thr)
        out[("q", thr)] = quality(f, tm, tz["tf"], tz["tt"])
        for w, hold, keep in settings:
            ev, kp = evidence(n, f, tm, sc, w, hold)
            p = C.decode_ev(pr, spans, n, ev, keep=kp if keep else None, **C.RAW)
            p2 = C.decode_ev(pf, spans, n, ev, keep=kp if keep == 2 else None, **C.FOLDED)
            out[(thr, w, hold, keep)] = C.evaluate(name, p, p2)
    return (fold, name), out


def matches(tag, kind):
    import glob
    out = []
    for fold in C.PROVIDERS:
        for f in sorted(glob.glob(os.path.join(C.RUNS, tag, fold, "*.npz"))):
            if str(np.load(f)["kind"]) == kind:
                out.append((fold, os.path.basename(f)[:-4]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--kind", default="test")
    ap.add_argument("--v", type=int, default=1)
    ap.add_argument("--thr", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--w", nargs="+", type=float, default=[2, 4, 8])
    ap.add_argument("--hold", nargs="+", type=float, default=[1, 2])
    ap.add_argument("--keep", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    settings = [(w, h, k) for w in a.w for h in a.hold for k in a.keep]
    ms = matches(a.tag, a.kind)
    res = dict(C.pool_map(one, [(fo, n, a.tag, a.touch, a.v, settings, a.thr) for fo, n in ms], a.workers))
    # val: a match can sit in two folds; key the report by fold+name, provider prefix first
    R = {f"{n}@{fo}": r for (fo, n), r in res.items()}
    base = {n: r["base"] for n, r in R.items()}
    print(f"== {a.tag} + touches {a.touch}, {a.kind}, ball version {a.v}; IDSSE Metrica SkillCorner | mean")
    C.report("base", base)
    for thr in a.thr:
        qs = [np.sum([r[("q", thr)] for n, r in R.items() if n.startswith(pv)], 0) for pv in C.PROVIDERS]
        print(f"threshold {thr}: precision " + " ".join(f"{x[0] / max(x[1], 1):.2f}" for x in qs) + "  recall " + " ".join(f"{x[2] / max(x[3], 1):.2f}" for x in qs)
              + "  events per min " + " ".join(f"{x[1] / (sum(1 for n in R if n.startswith(pv)) * 95):.1f}" for x, pv in zip(qs, C.PROVIDERS)))
        for s in settings:
            C.report(f"thr{thr} w{s[0]} hold{s[1]} keep{s[2]}", {n: r[(thr,) + s] for n, r in R.items()}, base)


if __name__ == "__main__":
    main()
