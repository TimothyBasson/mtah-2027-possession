"""How much does a net's held-out score depend on the ball simulation being right?

    python alldata/robust/stress.py --build                 # feature rows under every condition, once
    python alldata/robust/stress.py --tags a0 lp0 mx0       # fold nets re-run under every condition

Each condition is one fixed way of turning an outside ball into a BAR-ATH-like one (the same
hidden stretches in all of them, except `seen`); the fold nets of each tag label their held-out
provider under it. Columns: spellF1 on the labels as built (head 1) and folded (head 2), mean of
the three providers; the players-only net does not see the ball, so it is the same on every row.
"""
import argparse
import glob
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
L = os.path.join(ROOT, "alldata", "learned")
sys.path[:0] = [L, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import feats  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "alldata", "stress")
# name: (filter, parameters, frames late, ball kept through every stoppage)
CONDS = {
    "lp 5 @0.70 (BAR-ATH fit)": ("lp", (5, 0.70), 0, False),
    "lp 5 @0.55 (smoother)": ("lp", (5, 0.55), 0, False),
    "lp 5 @0.45 (much smoother)": ("lp", (5, 0.45), 0, False),
    "lp 5 @0.90 (sharper)": ("lp", (5, 0.90), 0, False),
    "lp 5 @1.20 (much sharper)": ("lp", (5, 1.20), 0, False),
    "lp 2 @0.70 (gentler slope)": ("lp", (2, 0.70), 0, False),
    "lp 5 @0.70, 0.32 s late": ("lp", (5, 0.70), 8, False),
    "lp 5 @0.70, 0.32 s early": ("lp", (5, 0.70), -8, False),
    "lp 5 @0.70, seen in stoppages": ("lp", (5, 0.70), 0, True),
    "window 16 (first nets)": ("para", (16,), 0, False),
    "window 8": ("para", (8,), 0, False),
    "window 24": ("para", (24,), 0, False),
}


def slug(c):
    return "".join(ch if ch.isalnum() else "_" for ch in c)


def build_one(path):
    import build
    name = os.path.basename(path)[:-4]
    d = dict(np.load(path))
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    bt = build.bt_probs(name, len(halves))
    fps = int(d["fps"])
    for cond, (kind, par, late, seen) in CONDS.items():
        dst = os.path.join(OUT, slug(cond), name + ".npz")
        if os.path.exists(dst):
            continue
        rng = np.random.default_rng(zlib.crc32(f"{name} stress".encode()))  # the same gaps under every condition
        bxy = d["ball"][:, :2].astype(np.float32).copy()
        o = {}
        for i, (a, b) in enumerate(halves):
            x = feats.simulate_visibility(feats.hide_frozen(bxy[a:b + 1], fps), d["labels"][a:b + 1], rng, fps, 0.0 if seen else None)
            x = feats.lowpass(x, par[0], par[1], fps) if kind == "lp" else feats.smooth(x, par[0] * fps / 25)
            bxy[a:b + 1] = feats.shift(x, late)
            o[f"F{i}"] = feats.half_features(d["players"], bxy, d["team"], a, b, d["pitch"], fps, extra=[("bt", bt[i])])[0]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez(dst + ".tmp.npz", names=np.array(feats.NAMES), **o)
        os.replace(dst + ".tmp.npz", dst)
    return name


def score_one(args):
    tag, fold = args
    import torch
    import fit as T
    import score as S
    import decode
    import possession_metrics as pm
    torch.set_num_threads(1)
    ck = torch.load(os.path.join(T.CACHE, "runs", tag, fold, "net.pt"), map_location="cpu")
    net = T.Net(len(ck["names"]), ck["d"])
    net.load_state_dict(ck["sd"])
    sym = feats.symmetry(ck["names"])
    res = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "external", fold + "_*.npz"))):
        name = os.path.basename(path)[:-4]
        raw, fo, spans = S.truths(name)
        for cond in CONDS:
            z = np.load(os.path.join(OUT, slug(cond), name + ".npz"))
            names = [str(n) for n in z["names"]]
            cols = [names.index(n) for n in ck["names"]]
            q = [T.predict(net, z[f"F{i}"][:, cols], sym) for i in range(len(spans))]
            r = []
            for t, pr in ((raw, [x[:, :3] for x in q]), (fo, [x[:, 3:] for x in q])):
                pred = pm.label_codes(decode.labels_from(pr, spans, len(t), **S.RAW))
                r.append(pm.evaluate(t, pred, bootstrap_n=0)["spell_f1_mean"])
            res.setdefault(cond, []).append(r)
    print("scored", tag, fold, file=sys.stderr, flush=True)
    return tag, fold, {c: np.mean(v, 0) for c, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    folds = ("idsse", "metrica", "skillcorner")
    with ProcessPoolExecutor(a.jobs) as ex:
        if a.build:
            for n in ex.map(build_one, sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))):
                print(n, flush=True)
        if not a.tags:
            return
        res = {(t, f): r for t, f, r in ex.map(score_one, [(t, f) for t in a.tags for f in folds])}
    print(f"{'ball shown':32s}" + "".join(f"{t:>16s}" for t in a.tags))
    for c in CONDS:
        m = [np.mean([res[(t, f)][c] for f in folds], 0) for t in a.tags]
        print(f"{c:32s}" + "".join(f"   {x[0]:.3f} / {x[1]:.3f}" for x in m))
    for f in folds:
        c = next(iter(CONDS))
        print(f"{'  ' + f + ', first row':32s}" + "".join(f"   {res[(t, f)][c][0]:.3f} / {res[(t, f)][c][1]:.3f}" for t in a.tags))


if __name__ == "__main__":
    main()
