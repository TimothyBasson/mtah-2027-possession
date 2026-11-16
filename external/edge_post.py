"""Stoppage-edge post-processing tests (restart hold, folding short possessions between two
stoppages) on the saved v9 held-out decodes (data/cache/edge, from edge_diag.py)."""
import os, sys, glob
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation")); sys.path.insert(0, os.path.join(ROOT, "external"))
import score_hmm as sh
import hmm.decode as dec

def ball_speed(ball, fps, smooth=0.4):
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    v = np.hypot(np.gradient(bx), np.gradient(by)) * fps
    w = max(1, int(round(smooth * fps)))
    # running median-ish: mean over window, NaN-aware
    k = np.ones(w)
    ok = np.isfinite(v)
    num = np.convolve(np.where(ok, v, 0), k, "same"); den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > w / 2, num / np.maximum(den, 1), np.nan)

def runs(lab):
    out = []; s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append([lab[s], s, i]); s = i
    return out

def fold_sandwich(pred, fps, max_len=2.0):
    p = pred.copy(); r = runs(p)
    for a, b, c in zip(r, r[1:], r[2:]):
        if a[0] == "D" and c[0] == "D" and b[0] in "HA" and (b[2] - b[1]) < max_len * fps:
            p[b[1]:b[2]] = "D"
    return p

VARIANTS = {}
def run(path):
    z = np.load(path); fps = int(z["fps"]); pred = z["pred"]; fold = z["fold"]; ball = z["ball"]
    out = {}
    for name, f in VARIANTS.items():
        q = f(pred, ball, fps)
        q[z["raw"] == "X"] = "X"
        out[name] = sh.score(fold, q)[0]
    return os.path.basename(path)[:-4], out

def report(rows):
    names = list(next(iter(rows.values())))
    base = names[0]
    for m in names:
        per = []
        ups = []
        for k in sh.SETS:
            ns = [n for n in rows if n.startswith(k)]
            per.append(np.mean([rows[n][m] for n in ns]))
            ups.append(sum(rows[n][m] > rows[n][base] + 1e-9 for n in ns) - sum(rows[n][m] < rows[n][base] - 1e-9 for n in ns))
        print("%-34s %.3f / %.3f / %.3f  mean %.4f  net up %s" % (m, *per, np.mean(per), ups))

if __name__ == "__main__":
    import itertools
    VARIANTS["v9"] = lambda p, b, f: p.copy()
    for still, kick, ms in itertools.product([0.5, 1.0, 2.0], [3.0, 5.0], [0.5, 1.0, 2.0]):
        VARIANTS["hold s%.1f k%.0f m%.1f" % (still, kick, ms)] = (lambda s, k, m: lambda p, b, f: dec.hold_restarts(p, b[:, 0], b[:, 1], f, s, k, m))(still, kick, ms)
    for L in (1.5, 2.0, 3.0, 4.0):
        VARIANTS["sandwich %.1f" % L] = (lambda L: lambda p, b, f: fold_sandwich(p, f, L))(L)
    with Pool(4) as pool:
        rows = dict(pool.map(run, sorted(glob.glob(os.path.join(ROOT, "data/cache/edge/*.npz")))))
    report(rows)
