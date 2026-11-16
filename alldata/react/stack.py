"""Re-decide the base model's probabilities with the team reactions as extra evidence.

    python alldata/react/stack.py [--seeds 0 1] [--weight-brief 1] [--save tag]

A gradient-boosted model reads, per 0.2 s row, the base model's probabilities (now and 1, 2 and
4 s either side) and, in the `react` variant, the team-reaction rows; it gives new H/A/D
probabilities (one model taught on the labels as built, one on control-rule labels, averaged as
the base's two heads are), decoded like the submission. The `base only` variant has the same
form without the reaction rows: the difference between the two is what the reactions add.

Held out by provider: the corrector for a provider's matches is fitted on the other two
providers' held-out matches (each predicted by a base net that never saw its provider), so no
row is corrected by a model that saw its provider's labels.

--weight-brief w: rows inside true possessions under 5 s count w times in the fit.
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdata  # noqa: E402
import react as R  # noqa: E402
from common import PROVS, provider_mean  # noqa: E402

LAGS = (-20, -10, -5, 0, 5, 10, 20)


def base_cols(p):
    lp = np.log(np.clip(p, 1e-4, 1))
    T = len(lp)
    idx = np.arange(T)
    return np.concatenate([lp[np.clip(idx + k, 0, T - 1)] for k in LAGS], 1)


def swap_base(B):
    out = B.copy()
    out[:, 0::3], out[:, 1::3] = B[:, 1::3], B[:, 0::3]
    return out


def brief_mask(t, fps=25, max_s=5.0):
    """Frames inside H/A runs shorter than max_s."""
    e = np.flatnonzero(np.diff(t) != 0) + 1
    s, u = np.r_[0, e], np.r_[e, len(t)]
    out = np.zeros(len(t), bool)
    for a, b in zip(s, u):
        if t[a] < 2 and b - a < max_s * fps:
            out[a:b] = True
    return out


def table(m, use_react):
    """Per half: rows (X, X with the teams swapped)."""
    out = []
    for p, r in zip(m.probs, m.R):
        B = base_cols(p)
        out.append((np.hstack([B, r]), np.hstack([swap_base(B), R.swap(r)])) if use_react else (B, swap_base(B)))
    return out


def fit(ms, truth_of, use_react, seed, wb, sub=2):
    X, y, w = [], [], []
    for m in ms:
        t = truth_of(m)
        bm = brief_mask(t)
        for (x, xs), fr in zip(table(m, use_react), m.fr):
            lab = t[fr]
            k = np.flatnonzero(lab < 3)[seed % sub::sub]
            sw = np.where(lab[k] < 2, 1 - lab[k], lab[k])
            X += [x[k], xs[k]]
            y += [lab[k], sw]
            w += [np.where(bm[fr][k], wb, 1.0)] * 2
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.7, min_child_samples=200, random_state=seed, n_jobs=4, verbose=-1
                              ).fit(np.concatenate(X), np.concatenate(y), sample_weight=np.concatenate(w))


def apply(clfs, m, use_react):
    out = []
    for x, xs in table(m, use_react):
        q = 0
        for c in clfs:
            a, b = c.predict_proba(x), c.predict_proba(xs)
            q = q + (a + b[:, [1, 0, 2]]) / 2
        out.append(q / len(clfs))
    return out


def slice_counts(m, pred):
    out = []
    for t in (m.raw, m.folded):
        s = rdata.brief_slice(m, pred, t)
        out += [len(s), s[:, 3].sum(), s[:, 4].sum()]
    return out


def report(title, ms, preds):
    names = [m.name for m in ms]
    sc = provider_mean(names, [m.score(p) for m, p in zip(ms, preds)])
    cnt = np.array([slice_counts(m, p) for m, p in zip(ms, preds)], float)
    tot = np.array([cnt[[n.startswith(pv) for n in names]].sum(0) for pv in PROVS])
    line = f"{title:26s}"
    for j, lab in ((0, "built"), (1, "folded")):
        line += f" {lab} " + " ".join(f"{x:.3f}" for x in sc[:, j])
    f = tot[:, 4] / tot[:, 3]
    o = tot[:, 5] / tot[:, 3]
    line += "  brief found " + " ".join(f"{x:.2f}" for x in f) + "  to other team " + " ".join(f"{x:.2f}" for x in o)
    print(line, flush=True)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--weight-brief", type=float, default=1.0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    ms = rdata.load_all()
    print("columns: spellF1 IDSSE Metrica SkillCorner mean (labels as built, then control-rule); brief = true control-rule possessions under 5 s")
    report("base (pw1e)", ms, [m.decode() for m in ms])
    for use_react, title in ((False, "corrector, base only"), (True, "corrector, with reactions")):
        allp = []
        for seed in a.seeds:
            probs = {}
            for prov in PROVS:
                tr = [m for m in ms if m.fold != prov]
                clfs = [fit(tr, f, use_react, seed, a.weight_brief) for f in (lambda m: m.raw, lambda m: m.folded)]
                for m in (m for m in ms if m.fold == prov):
                    probs[m.name] = apply(clfs, m, use_react)
            report(f"{title} s{seed}", ms, [m.decode(probs[m.name]) for m in ms])
            allp.append(probs)
        avg = {n: [np.mean([p[n][i] for p in allp], 0) for i in range(len(allp[0][n]))] for n in allp[0]}
        report(f"{title} avg", ms, [m.decode(avg[m.name]) for m in ms])
        if a.save and use_react:
            d = os.path.join(R.OUT, a.save)
            os.makedirs(d, exist_ok=True)
            for n, p in avg.items():
                np.savez_compressed(os.path.join(d, n + ".npz"), **{f"p{i}": x.astype(np.float32) for i, x in enumerate(p)})


if __name__ == "__main__":
    main()
