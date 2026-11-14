"""Team naming per possession, on top of a base model's probabilities.

    python team/pieces.py --base v2 [--extra bt,bts1]

1. The base model's H/A/D probabilities are cut a little finer than its shipped decode.
2. Each piece (one possession as the base sees it) gets features of the whole piece: how sure and
   how steady the base was, what stands either side, and the two teams' shape and movement.
3. A gradient-boosted classifier says whether the base named the right team; if not, the piece is
   given to the other team. Pieces are merged and the 2 s minimum applied again.

The classifier for a held-out provider learns only from the other two providers' matches, each
seen through the base net that never trained on that match's provider, so what it learns from is
the kind of output it will meet. Players only: every feature comes from the base probabilities
and the players' tracking.
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import decode  # noqa: E402

CUT = dict(cost=1, min_team=1, min_dead=3)
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
LET = np.array(list("HADX"))


def shape_features(name):
    """Per 5 fps frame and half: [centroid x, forward speed, width, depth, tracked] for home and away,
    in the frame where home attacks +x. From the players alone."""
    cache = os.path.join(C.ROOT, "data", "cache", "team", "shape", name + ".npy")
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)
    import data as D
    d = np.load(os.path.join(C.ROOT, "data", "external", name + ".npz"))
    out = []
    for a, b in d["halves"]:
        X, M, _, _ = D.half_tensors(d["players"], d["ball"], d["team"], int(a), int(b), tuple(d["pitch"]), int(d["fps"]))
        f = []
        for sign in (1, -1):
            m = M & (X[..., 6] * sign > 0)
            n = np.maximum(m.sum(1), 1)
            mean = lambda v: (v * m).sum(1) / n  # noqa: E731
            cx, cy = mean(X[..., 0]), mean(X[..., 1])
            f += [cx, mean(X[..., 2]), np.sqrt(mean((X[..., 1] - cy[:, None]) ** 2)),
                  np.sqrt(mean((X[..., 0] - cx[:, None]) ** 2)), m.sum(1) / 11]
        out.append(np.stack(f, 1).astype(np.float32))
    arr = np.empty(len(out), object)
    arr[:] = out
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, arr, allow_pickle=True)
    return arr


def pieces(args):
    name, base, extra = args
    raw, fold, spans = C.truth(name)
    probs = [np.mean([C.stage1(t, name)[i] for t in base], 0) for i in range(len(spans))]
    others = [C.stage1(t, name) for t in extra]
    shape = shape_features(name)
    pred = decode.labels_from(probs, spans, len(raw), **CUT)
    rows, where = [], []
    for hi, (a, b) in enumerate(spans):
        p = probs[hi]
        lo = np.log(np.clip(p, 1e-4, 1))
        marg = lo[:, 0] - lo[:, 1]                       # home minus away, 5 fps
        rr = C.runs(pred[a:b + 1])
        for i, (s, e, v) in enumerate(rr):
            if v not in ("H", "A"):
                continue
            sg = 1.0 if v == "H" else -1.0
            s5, e5 = s // 5, max(s // 5 + 1, e // 5)
            m = sg * marg[s5:e5]
            n = len(m)
            f = [np.log((e - s) / 25), m.mean(), m.min(), np.percentile(m, 10), m.std(), (m > 0).mean(),
                 m[:5].mean(), m[-5:].mean(), m[n // 3:max(n // 3 + 1, 2 * n // 3)].mean(), p[s5:e5, 2].mean(),
                 sg * marg[max(0, s5 - 10):s5].mean() if s5 > 0 else 0.0,
                 sg * marg[e5:e5 + 10].mean() if e5 < len(marg) else 0.0]
            for j in (i - 2, i - 1, i + 1, i + 2):       # neighbours: same team +1, other team -1, dead 0
                if 0 <= j < len(rr):
                    js, je, jv = rr[j]
                    f += [0.0 if jv not in ("H", "A") else (1.0 if jv == v else -1.0), np.log((je - js) / 25)]
                else:
                    f += [0.0, -3.0]
            # the live stretch this piece sits in
            j0 = i
            while j0 > 0 and rr[j0 - 1][2] in ("H", "A"):
                j0 -= 1
            j1 = i
            while j1 < len(rr) - 1 and rr[j1 + 1][2] in ("H", "A"):
                j1 += 1
            f += [(s - rr[j0][0]) / 25, (rr[j1][1] - e) / 25, j1 - j0 + 1, i - j0, j1 - i]
            for o in others:
                om = sg * (np.log(np.clip(o[hi][s5:e5, 0], 1e-4, 1)) - np.log(np.clip(o[hi][s5:e5, 1], 1e-4, 1)))
                f += [om.mean(), (om > 0).mean()]
            sh = shape[hi][s5:e5]
            own, opp = (sh[:, :5], sh[:, 5:]) if v == "H" else (sh[:, 5:], sh[:, :5])
            f += [sg * own[:, 0].mean(), sg * opp[:, 0].mean(), sg * own[:, 1].mean(), sg * opp[:, 1].mean(),
                  own[:, 2].mean(), opp[:, 2].mean(), own[:, 3].mean(), opp[:, 3].mean(), own[:, 4].mean(), opp[:, 4].mean(),
                  sg * (own[-1, 0] - own[0, 0]), sg * (opp[-1, 0] - opp[0, 0]),
                  own[:, 2].mean() - opp[:, 2].mean(), own[:, 3].mean() - opp[:, 3].mean()]
            t = fold[a + s:a + e]
            nh, na = np.sum(t == "H"), np.sum(t == "A")
            right = np.nan if nh + na < 0.3 * (e - s) else float((nh >= na) == (v == "H"))
            rows.append(f + [right])
            where.append((a + s, a + e, CODE[v]))
    return name, np.array(rows, np.float32), np.array(where), pred


def relabel(pred, spans, where, flip):
    out = np.array([CODE[c] for c in pred], np.int8)
    for (s, e, k), fl in zip(where, flip):
        if fl:
            out[s:e] = 1 - k
    for a, b in spans:
        out[a:b + 1] = decode.absorb(out[a:b + 1], {0: 50, 1: 50, 2: 75, 3: 0})
    return LET[out]


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v2")
    ap.add_argument("--extra", default="")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    base, extra = a.base.split(","), [t for t in a.extra.split(",") if t]
    names = [os.path.basename(p)[:-4] for p in C.paths()]
    with Pool(2) as pool:
        P = pool.map(pieces, [(n, base, extra) for n in names])
    prov = np.array([n.split("_")[0] for n in names])
    res, imp = [], 0
    for held in C.PROVIDERS:
        tr = [p for p, pv in zip(P, prov) if pv != held]
        X = np.concatenate([p[1][:, :-1] for p in tr])
        y = np.concatenate([p[1][:, -1] for p in tr])
        w = np.concatenate([np.full(len(p[1]), 1.0 / sum(len(q[1]) for q in tr if q[0].split("_")[0] == p[0].split("_")[0])) for p in tr])
        ok = np.isfinite(y)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)
        clf.fit(X[ok], y[ok], sample_weight=w[ok] * ok.sum() / w[ok].sum())
        imp = imp + clf.feature_importances_
        for name, F, where, pred in [p for p, pv in zip(P, prov) if pv == held]:
            raw, fold, spans = C.truth(name)
            pr = clf.predict_proba(F[:, :-1])[:, 1]
            y_ = F[:, -1]
            g = np.isfinite(y_)
            row = [C.spell_f1(fold, decode.labels_from([np.mean([C.stage1(t, name)[i] for t in base], 0) for i in range(len(spans))], spans, len(raw), **C.DECODE)),
                   C.spell_f1(fold, relabel(pred, spans, where, np.zeros(len(pr), bool))),
                   C.spell_f1(fold, relabel(pred, spans, where, pr < 0.5)),
                   C.spell_f1(fold, relabel(pred, spans, where, pr < 0.4)),
                   C.spell_f1(fold, relabel(pred, spans, where, np.where(g, y_ == 0, False))),
                   y_[g].mean(), ((pr[g] > 0.5) == (y_[g] > 0.5)).mean()]
            res.append((name, row))
    names2 = [r[0] for r in res]
    T = np.array(C.provider_mean(names2, [r[1] for r in res]))
    print(f"base {a.base}  extra {a.extra or '-'}                    idsse  metrica  skillc   mean")
    for j, k in enumerate(("base, shipped folded decode", "fine cut, 2 s min", "pieces renamed at 0.5", "pieces renamed at 0.4",
                           "pieces renamed by the truth (ceiling)", "share of pieces the base names right", "classifier right about the base")):
        print(f"  {k:40s} " + "  ".join(f"{x:.3f}" for x in T[:, j]))
    print("top features (index: gain count)", np.argsort(-imp)[:12], np.sort(-imp)[:12] * -1)
    if a.save:
        with open(a.save, "wb") as f:
            pickle.dump(res, f)


if __name__ == "__main__":
    main()
