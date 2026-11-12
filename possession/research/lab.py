"""Held-out-by-provider experiments on the cached 5 Hz features (feats.py).

    uv run python possession/research/lab.py
"""
import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "possession")]
import feats  # noqa: E402
import possession_metrics as pm  # noqa: E402

HZ = feats.HZ
PROVIDERS = ("idsse", "metrica", "skillcorner")


def load():
    ms = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz"))):
        r = feats.cached(p)
        r["name"] = os.path.basename(p)[:-4]
        r["prov"] = r["name"].split("_")[0]
        ms.append(r)
    return ms


def per_half(fn, x, half):
    out = np.full(len(x), np.nan)
    for h in np.unique(half):
        m = half == h
        out[m] = fn(x[m])
    return out


def smooth(v, sec):
    n = max(1, int(round(sec * HZ)))
    k = np.ones(2 * n + 1)
    ok = np.isfinite(v)
    c = np.convolve(ok, k, "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, np.convolve(np.where(ok, v, 0), k, "same") / c, np.nan)


def side(v, sec, ahead):
    n = max(1, int(round(sec * HZ)))
    c, k = np.r_[0, np.cumsum(np.nan_to_num(v))], np.r_[0, np.cumsum(np.isfinite(v))]
    i = np.arange(len(v))
    lo, hi = (i + 1, np.minimum(len(v), i + 1 + n)) if ahead else (np.maximum(0, i - n), i)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c[hi] - c[lo]) / (k[hi] - k[lo])


def expand(m, group, names, views):
    """Columns for the named base features under the named temporal views."""
    cols, labels = [], []
    for nm in names:
        x = m[group][nm]
        for vw in views:
            kind, sec = vw[0], float(vw[1:])
            if kind == "s":
                c = per_half(lambda v: smooth(v, sec), x, m["half"])
            elif kind == "d":   # after minus before
                c = per_half(lambda v: side(v, sec, True) - side(v, sec, False), x, m["half"])
            elif kind == "a":
                c = per_half(lambda v: side(v, sec, True), x, m["half"])
            elif kind == "b":
                c = per_half(lambda v: side(v, sec, False), x, m["half"])
            cols.append(c); labels.append(f"{nm}:{vw}")
    return np.column_stack(cols), labels


def logit_fit(X, y, wt, l2=1e-3, intercept=True, iters=30):
    Z = np.c_[np.ones(len(X)), X] if intercept else X
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (wt * (p - y)) + l2 * w
        hess = (Z * (wt * p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def absorb(labels, min_len):
    runs = [[k, len(list(g))] for k, g in itertools.groupby(labels.tolist())]
    while len(runs) > 1:
        short = [i for i, (k, n) in enumerate(runs) if n < min_len.get(k, 0)]
        if not short:
            break
        i = min(short, key=lambda i: runs[i][1])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: runs[j][1])
        runs[j][1] += runs.pop(i)[1]
        if 0 < i < len(runs) and runs[i - 1][0] == runs[i][0]:
            runs[i - 1][1] += runs.pop(i)[1]
    return np.array([k for k, n in runs for _ in range(n)])


def decode(m, home_logit, dead_logit, min_team=2.0, min_dead=3.0, min_alive=2.0):
    """The current model's clean-up, at 5 Hz, then held to 25 fps. Returns the (n25,) labels."""
    lab = np.full(len(m["labels25"]), "X", dtype="<U1")
    for h, (a, b) in enumerate(m["halves"]):
        sel = m["half"] == h
        home = absorb(np.nan_to_num(home_logit[sel]) >= 0, {True: min_team * HZ, False: min_team * HZ})
        alive = absorb(np.nan_to_num(dead_logit[sel]) < 0, {True: min_alive * HZ, False: min_dead * HZ})
        out5 = np.where(alive, np.where(home, "H", "A"), "D")
        n = b + 1 - a
        lab[a:b + 1] = np.repeat(out5, feats.STEP)[:n]
    return lab


def score(m, pred25):
    r = pm.evaluate(pm.label_codes(m["labels25"]), pm.label_codes(pred25), bootstrap_n=0)
    return np.array([r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]])


def design(ms, group, names, views, cache={}):
    key = (group, tuple(names), tuple(views))
    if key not in cache:
        cache[key] = [expand(m, group, names, views) for m in ms]
    return [c[0] for c in cache[key]], cache[key][0][1]


def fit_fold(ms, Xs, test, target, sym, l2=1e-3):
    """Logistic fitted on the providers other than `test`, each provider weighted equally.
    target 'who': home vs away on live frames (symmetric: no intercept, scale only).
    target 'dead': dead vs alive on graded frames."""
    Xl, yl, wl = [], [], []
    count = {p: sum(m["prov"] == p for m in ms) for p in PROVIDERS}
    for m, X in zip(ms, Xs):
        if m["prov"] == test:
            continue
        lab = m["labels"]
        g = np.isin(lab, ["H", "A"]) if target == "who" else np.isin(lab, ["H", "A", "D"])
        g &= np.isfinite(X).all(1)
        Xl.append(X[g]); yl.append((lab[g] == ("H" if target == "who" else "D")).astype(float))
        wl.append(np.full(g.sum(), 1 / (count[m["prov"]] * max(1, g.sum()))))
    X, y, wt = np.vstack(Xl), np.concatenate(yl), np.concatenate(wl)
    wt /= wt.sum()
    if sym:
        mu = np.zeros(X.shape[1]); sd = np.sqrt((X ** 2).T @ wt)
    else:
        mu = X.T @ wt; sd = np.sqrt(((X - mu) ** 2).T @ wt)
    sd[sd == 0] = 1
    w = logit_fit((X - mu) / sd, y, wt, l2=l2, intercept=not sym)
    return dict(w=w, mu=mu, sd=sd, sym=sym)


def apply(fit, X):
    Z = np.nan_to_num((X - fit["mu"]) / fit["sd"])
    return Z @ fit["w"] if fit["sym"] else fit["w"][0] + Z @ fit["w"][1:]


def run(ms, who=None, dead=None, verbose=True, tag="", **dec):
    """who / dead: (names, views). None uses the baseline rule for that part (spread lean for who,
    the three shipped stoppage signals for dead). Prints and returns the held-out table."""
    who = who or (["spread"], ["s2"])
    dead = dead or (["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])
    Xw, lw = design(ms, "who", *who)
    Xd, ld = design(ms, "dead", *dead)
    rows = {}
    facc = {}
    for test in PROVIDERS:
        fw = fit_fold(ms, Xw, test, "who", True)
        fd = fit_fold(ms, Xd, test, "dead", False)
        sc, fa = [], []
        for m, xw, xd in zip(ms, Xw, Xd):
            if m["prov"] != test:
                continue
            hl, dl = apply(fw, xw), apply(fd, xd)
            sc.append(score(m, decode(m, hl, dl, **dec)))
            lab = m["labels"]; live = np.isin(lab, ["H", "A"]); gr = lab != "X"
            fa.append([((hl[live] >= 0) == (lab[live] == "H")).mean(), ((dl[gr] >= 0) == (lab[gr] == "D")).mean()])
        rows[test] = np.mean(sc, 0); facc[test] = np.mean(fa, 0)
    mean = np.mean(list(rows.values()), 0); fm = np.mean(list(facc.values()), 0)
    if verbose:
        print(f"{tag:42s} spellF1 {mean[0]:.3f} ({' '.join(f'{rows[p][0]:.3f}' for p in PROVIDERS)})  macro {mean[1]:.3f}"
              f"  a/d {mean[2]:.3f}  team {mean[3]:.3f} | frame who {fm[0]:.3f} dead {fm[1]:.3f}", flush=True)
    return mean, rows, fm


if __name__ == "__main__":
    ms = load()
    V = ["s1", "s3", "d2"]
    run(ms, tag="baseline-like (spread s2, 3 dead signals)")
    run(ms, who=(["spread"], V), tag="spread, 3 views")
    run(ms, who=(["wid", "len", "vcx"], V), tag="width+length+centroid velocity (other thread)")
    base = ["wid", "len", "vcx"]
    for nm in ms[0]["who"]:
        if nm in base:
            continue
        run(ms, who=(base + [nm], V), tag=f"  + {nm}")
    run(ms, who=(list(ms[0]["who"]), V), tag="all who features")
