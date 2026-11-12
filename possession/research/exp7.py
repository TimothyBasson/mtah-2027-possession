"""Self-training per match (no labels of that match): refit the who model on its own confident
frames, shrunk toward the weights fitted on the other providers."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, lab
from lab import *
ms = load()
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
Xw, _ = design(ms, "who", list(ms[0]["who"]), ["s1", "s3", "d2"])
names = list(ms[0]["dead"])
B = design(ms, "dead", ["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])[0]
R = design(ms, "dead", [n for n in names if n not in ("sp_med", "acc_p80")], ["s1", "s3", "d2"])[0]
Xd = [np.c_[a, b] for a, b in zip(B, R)]

def adapt(Z, logit0, w0, keep, lam, rounds=2, soft=False):
    """Z standardised (n,k). Returns logits from weights refitted on pseudo-labels."""
    w = np.r_[0.0, w0]; A = np.c_[np.ones(len(Z)), Z]; logit = logit0
    prior = np.r_[0.0, w0]
    for _ in range(rounds):
        thr = np.quantile(np.abs(logit), 1 - keep)
        s = np.abs(logit) >= thr
        y = (logit[s] > 0).astype(float); As = A[s]
        n = s.sum()
        for _ in range(15):
            p = 1 / (1 + np.exp(-np.clip(As @ w, -30, 30)))
            g = As.T @ (p - y) / n + lam * (w - prior)
            H = (As * (p * (1 - p))[:, None]).T @ As / n + lam * np.eye(len(w))
            w -= np.linalg.solve(H, g)
        logit = A @ w
    return logit

for keep, lam in [(0, 0), (0.5, 0.3), (0.5, 0.1), (0.5, 0.03), (0.3, 0.1), (0.7, 0.1), (0.5, 0.01)]:
    rows, facc = [], []
    for test in PROVIDERS:
        fw = fit_fold(ms, Xw, test, "who", True); fd = fit_fold(ms, Xd, test, "dead", False)
        sc, fa = [], []
        for m, xw, xd in zip(ms, Xw, Xd):
            if m["prov"] != test: continue
            Z = np.nan_to_num((xw - fw["mu"]) / fw["sd"]); hl = Z @ fw["w"]; dl = apply(fd, xd)
            if keep:
                alive = dl < 0          # adapt on frames the stoppage model calls live
                hl2 = hl.copy(); hl2[alive] = adapt(Z[alive], hl[alive], fw["w"], keep, lam); 
                hl2[~alive] = adapt(Z, hl, fw["w"], keep, lam)[~alive] if False else hl[~alive]
                hl = hl2
            sc.append(score(m, decode(m, hl, dl)))
            lab_ = m["labels"]; live = np.isin(lab_, ["H", "A"])
            fa.append(((hl[live] >= 0) == (lab_[live] == "H")).mean())
        rows.append(np.mean(sc, 0)); facc.append(np.mean(fa))
    r = np.array(rows)
    print(f"keep {keep} lam {lam}: spellF1 {r[:,0].mean():.3f} ({r[:,0].round(3)}) team {r[:,3].mean():.3f} frame who {np.mean(facc):.3f}", flush=True)
