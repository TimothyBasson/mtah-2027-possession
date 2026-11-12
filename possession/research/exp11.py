"""Who is on the ball, as a discrete choice among the players (conditional logit on hand-made
per-player features), then each team's summed probability as a 'who' signal."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats2, feats3, lab
from lab import *
from concurrent.futures import ProcessPoolExecutor
OUT = os.path.join(ROOT, "data", "cache", "research3_scores")
paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
NF = len(feats3.NAMES)

def train_rows(path, every=3, cap=22):
    f, tgt, team = feats3.cached(path)
    lab5 = feats.cached(path)["labels"]
    rows = np.flatnonzero((tgt >= 0) & np.isin(lab5, ["H", "A"]))[::every]
    X = np.zeros((len(rows), cap, NF), np.float32); M = np.zeros((len(rows), cap), bool); y = np.zeros(len(rows), int)
    for r, i in enumerate(rows):
        slots = np.flatnonzero(np.isfinite(f[i, :, 0]))[:cap]
        if tgt[i] not in slots: y[r] = -1; continue
        X[r, :len(slots)] = np.nan_to_num(f[i, slots]); M[r, :len(slots)] = True; y[r] = int(np.flatnonzero(slots == tgt[i])[0])
    k = y >= 0
    return X[k], M[k], y[k]

def fit_choice(data, l2=1e-3):
    """data: [(prov, X, M, y)]. Returns mu, sd, w."""
    count = {}
    for p, *_ in data: count[p] = count.get(p, 0) + 1
    X = np.concatenate([d[1] for d in data]); M = np.concatenate([d[2] for d in data]); y = np.concatenate([d[3] for d in data])
    wt = np.concatenate([np.full(len(d[3]), 1 / (count[d[0]] * len(d[3]))) for d in data]); wt /= wt.sum()
    flat = X[M]; mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Z = np.where(M[..., None], (X - mu) / sd, 0).astype(np.float64)
    T = Z[np.arange(len(y)), y]
    w = np.zeros(NF)
    for it in range(20):
        s = Z @ w; s[~M] = -1e9; s -= s.max(1, keepdims=True)
        P = np.exp(s); P /= P.sum(1, keepdims=True)
        Ef = np.einsum("nk,nkf->nf", P, Z)
        g = (wt[:, None] * (T - Ef)).sum(0) - l2 * w
        H = np.einsum("n,nk,nkf,nkg->fg", wt, P, Z, Z) - np.einsum("n,nf,ng->fg", wt, Ef, Ef) + l2 * np.eye(NF)
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-5: break
    acc = (P.argmax(1) == y).mean()
    return dict(mu=mu, sd=sd, w=w, train_top1=acc)

def apply_choice(path, fit):
    f, tgt, team = feats3.cached(path)
    ok = np.isfinite(f[..., 0])
    s = np.nan_to_num((f - fit["mu"]) / fit["sd"]).astype(np.float64) @ fit["w"]
    s[~ok] = -1e9; s -= s.max(1, keepdims=True)
    P = np.exp(s); P /= P.sum(1, keepdims=True)
    qh = P[:, team == 0].sum(1)
    lean = np.clip(np.log((qh + 1e-4) / (1 - qh + 1e-4)), -8, 8)
    lean[~ok.any(1)] = np.nan
    ent = -(P * np.log(P + 1e-12)).sum(1)
    top = P.argmax(1)
    has = tgt >= 0
    return dict(lean=lean, ent=ent, pmax=P.max(1), top1=(top[has] == tgt[has]).mean(), team_top=(team[top[has]] == team[tgt[has]]).mean(), P=P.astype(np.float32))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(train_rows, paths))
    prov = [os.path.basename(p).split("_")[0] for p in paths]
    ms = load()
    for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
    for m in ms: m["who"]["ball_lean"] = np.full(len(m["idx"]), np.nan); m["dead"]["pb_ent"] = np.full(len(m["idx"]), np.nan); m["dead"]["pb_pmax"] = np.full(len(m["idx"]), np.nan)
    for test in PROVIDERS:
        fit = fit_choice([(pv, *r) for pv, r in zip(prov, rows) if pv != test])
        order = np.argsort(-np.abs(fit["w"]))
        print(test, "train top-1", round(float(fit["train_top1"]), 3), "weights:", ", ".join(f"{feats3.NAMES[i]} {fit['w'][i]:+.2f}" for i in order[:12]))
        t1, tt = [], []
        for m, p, pv in zip(ms, paths, prov):
            if pv != test: continue
            r = apply_choice(p, fit)
            m["who"]["ball_lean"] = r["lean"]; m["dead"]["pb_ent"] = r["ent"]; m["dead"]["pb_pmax"] = r["pmax"]
            t1.append(r["top1"]); tt.append(r["team_top"])
        print(f"  held out on {test}: top-1 player {np.mean(t1):.3f}, top-1 player's team right {np.mean(tt):.3f} (frames with the ball within 2 m of a player)", flush=True)
    np.savez_compressed(os.path.join(OUT, "leans.npz"), **{m["name"]: np.c_[m["who"]["ball_lean"], m["dead"]["pb_ent"], m["dead"]["pb_pmax"]] for m in ms})
    V = ["s1", "s3", "d2"]
    WHO = ["wid", "len", "vcx", "speed", "press_max", "pol"]
    run(ms, who=(["ball_lean"], ["s0.4", "s1", "s3", "d2"]), tag="ball_lean alone")
    run(ms, who=(WHO, V), tag="who6")
    run(ms, who=(WHO + ["ball_lean"], V), tag="who6 + ball_lean")
    def put(tag, Xs):
        k = ("who", ("custom", tag), ("x",)); design.__defaults__[0][k] = [(x, ["c"]) for x in Xs]
        return run(ms, who=(("custom", tag), ("x",)), tag=tag)
    A, _ = design(ms, "who", WHO, V); Bm, _ = design(ms, "who", ["ball_lean"], ["s0.4", "s1", "s3", "d2", "a1", "b1"])
    put("who6 + ball_lean (6 views)", [np.c_[a, b] for a, b in zip(A, Bm)])
