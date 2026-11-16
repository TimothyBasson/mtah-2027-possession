"""How much do the two teams' reactions say about who has the ball, and do they say it where the
base model is wrong?

    python alldata/react/measure.py [--seed 0]

A gradient-boosted classifier reads only the reaction rows (react.py) and names the team in
possession on live frames (control-rule labels), fitted on two providers and applied to the
third. Reported per held-out provider: how often it names the right team on all live frames,
inside brief possessions (under 5 s) the base found, and inside brief possessions the base gave
to the other team (failure mode 1); beside the base model's own probabilities there.
Saves the classifier's probability of Home per row: data/cache/alldata/react/clf_s<seed>/<match>.npz
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
import stack as S  # noqa: E402
from common import PROVS  # noqa: E402


BRIEF_ONLY = False


def rows(ms, sub=1):
    X, y = [], []
    for m in ms:
        bm = S.brief_mask(m.folded)
        for r, fr in zip(m.R, m.fr):
            t = m.folded[fr]
            k = np.flatnonzero((t < 2) & bm[fr])[::max(sub // 2, 1)] if BRIEF_ONLY else np.flatnonzero(t < 2)[::sub]
            X += [r[k], R.swap(r[k])]
            y += [t[k], 1 - t[k]]
    return np.concatenate(X), np.concatenate(y)


def fit(ms, seed, cols=None, sub=2):
    X, y = rows(ms, sub)
    if cols is not None:
        X = X[:, cols]
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.7, min_child_samples=200, random_state=seed, n_jobs=4, verbose=-1).fit(X, y)


def predict(clf, r, cols=None):
    """P(Home), made exactly symmetric under the team swap."""
    a, b = (r, R.swap(r)) if cols is None else (r[:, cols], R.swap(r)[:, cols])
    return (clf.predict_proba(a)[:, 0] + clf.predict_proba(b)[:, 1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ball", action="store_true", help="leave out the columns that read the ball (toball, behind)")
    ap.add_argument("--brief-only", action="store_true", help="fit on frames inside true possessions under 5 s only")
    a = ap.parse_args()
    global BRIEF_ONLY
    BRIEF_ONLY = a.brief_only
    ms = rdata.load_all()
    cols = None
    if a.no_ball:
        cols = [i for i, n in enumerate(R.NAMES) if not any(s in n for s in ("toball", "behind", "ball_seen"))]
    out = os.path.join(R.OUT, f"clf{'_nb' if a.no_ball else ''}{'_brief' if a.brief_only else ''}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    print("share of frames with the right team named (control-rule labels), reaction classifier | base model")
    print(f"{'':12s}{'all live':>16s}{'brief, found':>16s}{'brief, other team':>20s}{'n other':>9s}")
    imp = 0
    for prov in PROVS:
        clf = fit([m for m in ms if m.fold != prov], a.seed, cols)
        imp = imp + clf.feature_importances_
        acc = np.zeros((3, 3))
        nmiss = 0
        for m in (m for m in ms if m.fold == prov):
            ph = [predict(clf, r, cols) for r in m.R]
            np.savez_compressed(os.path.join(out, m.name + ".npz"), **{f"h{i}": p.astype(np.float32) for i, p in enumerate(ph)})
            n = len(m.raw)
            P = np.full(n, np.nan)   # reaction P(home), per frame
            B = np.full(n, np.nan)   # base P(home | live)
            for (h0, h1), p, q in zip(m.spans, ph, m.probs):
                P[h0:h1 + 1] = rdata.decode.upsample(np.stack([p, 1 - p, np.zeros_like(p)], 1), h1 - h0 + 1)[:, 0]
                qq = rdata.decode.upsample(q, h1 - h0 + 1)
                B[h0:h1 + 1] = qq[:, 0] / (qq[:, 0] + qq[:, 1])
            t = m.folded
            live = (t < 2) & np.isfinite(P)
            okr, okb = (P > 0.5) == (t == 0), (B > 0.5) == (t == 0)
            masks = [live, np.zeros(n, bool), np.zeros(n, bool)]
            for s, e, k, found, other in rdata.brief_slice(m, m.decode()):
                if found:
                    masks[1][s:e] = True
                elif other:
                    masks[2][s:e] = True
                    nmiss += 1
            for j, mk in enumerate(masks):
                mk = mk & live
                acc[j] += [okr[mk].sum(), okb[mk].sum(), mk.sum()]
        print(f"{prov:12s}" + "".join(f"{x[0] / x[2]:9.3f} |{x[1] / x[2]:5.3f}" + ("    " if j == 2 else "") for j, x in enumerate(acc)) + f"{nmiss:9d}")
    top = np.argsort(-imp)[:15]
    print("most used columns:", ", ".join((R.NAMES if cols is None else [R.NAMES[i] for i in cols])[i] for i in top))


if __name__ == "__main__":
    main()
