"""Which output to ship, and a stress test of the visibility simulation.

    python alldata/learned/heads.py ps1e            # each head against each label view
    python alldata/learned/heads.py s1 --always-seen  # fold nets re-run with the ball in view through every stoppage
    python alldata/learned/heads.py s1 hw=8           # ... with another smoothing window (half-width, frames)
    python alldata/learned/heads.py s1 shift=8        # ... with the ball 8 frames late (negative: early)

The simulation hides the ball inside stoppages, and it places those gaps with the labels, so a
model could score by reading the gaps. --always-seen is the other extreme: no stoppage hides the
ball (it is drawn between where it went out and where play restarts), so the gaps say nothing.
"""
import glob
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
import decode  # noqa: E402
import possession_metrics as pm  # noqa: E402

D = dict(cost=4, min_team=1, min_dead=1)


def always_seen(tag, q=0.0, hw=None, shift=0):
    import torch
    import feats
    import fit as T
    import build
    torch.set_num_threads(1)
    out = {}
    for fold in ("idsse", "metrica", "skillcorner"):
        ck = torch.load(os.path.join(T.CACHE, "runs", tag, fold, "net.pt"), map_location="cpu")
        net = T.Net(len(ck["names"]), ck["d"])
        net.load_state_dict(ck["sd"])
        sym = feats.symmetry(ck["names"])
        for path in sorted(glob.glob(os.path.join(S.ROOT, "data", "external", fold + "_*.npz"))):
            name = os.path.basename(path)[:-4]
            d = dict(np.load(path))
            halves = [tuple(int(v) for v in h) for h in d["halves"]]
            bt = build.bt_probs(name, len(halves))
            rng = np.random.default_rng(zlib.crc32(f"{name} always".encode()))
            bxy = d["ball"][:, :2].astype(np.float32).copy()
            probs = []
            for i, (a, b) in enumerate(halves):
                if hw is None and not shift:
                    bxy[a:b + 1] = feats.bar_like(bxy[a:b + 1], d["labels"][a:b + 1], rng, int(d["fps"]), q=q)
                else:  # another smoothing window (half-width in frames) and / or the ball late by `shift` frames
                    x = feats.simulate_visibility(feats.hide_frozen(bxy[a:b + 1]), d["labels"][a:b + 1], rng, int(d["fps"]), q)
                    x = feats.smooth(x, hw or rng.uniform(14.0, 18.0))
                    y = np.full_like(x, np.nan)
                    if shift > 0:
                        y[shift:] = x[:-shift]
                    elif shift < 0:
                        y[:shift] = x[-shift:]
                    else:
                        y = x
                    bxy[a:b + 1] = y
                F, _ = feats.half_features(d["players"], bxy, d["team"], a, b, d["pitch"], int(d["fps"]), extra=[("bt", bt[i])])
                cols = [feats.NAMES.index(n) for n in ck["names"]]
                probs.append(T.predict(net, F[:, cols], sym))
            out[name] = probs
    return out


def main():
    tag = sys.argv[1]
    res = {}
    opt = {k: float(v) for k, v in (x.split("=") for x in sys.argv[2:] if "=" in x)}
    stress = "--always-seen" in sys.argv or bool(opt)
    if stress:  # e.g. hw=8, hw=24, shift=8 (ball 0.32 s late), shift=-8; q=0 with --always-seen
        allp = always_seen(tag, q=0.0 if "--always-seen" in sys.argv else None,
                           hw=opt.get("hw"), shift=int(opt.get("shift", 0)))
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(tag), fold, fold + "_*.npz"))):
            name = os.path.basename(f)[:-4]
            raw, fo, spans = S.truths(name)
            z = np.load(f)
            for hk in ("built", "folded", "both"):
                if stress:
                    q = allp[name]
                    a, b = [x[:, :3] for x in q], [x[:, 3:] for x in q]
                else:
                    a, b = [z[f"s{i}"] for i in range(len(spans))], [z[f"fs{i}"] for i in range(len(spans))]
                probs = a if hk == "built" else b if hk == "folded" else [(x + y) / 2 for x, y in zip(a, b)]
                pred = pm.label_codes(decode.labels_from(probs, spans, len(raw), **D))
                for tn, t in (("as built", raw), ("folded", fo)):
                    r = pm.evaluate(t, pred, bootstrap_n=0)
                    res.setdefault((hk, tn, fold), []).append((r["spell_f1_mean"], r["alive_dead_f1"], r["team_acc_both_alive"]))
    for hk in ("built", "folded", "both"):
        for tn in ("as built", "folded"):
            per = np.array([np.mean(res[(hk, tn, f)], 0) for f in ("idsse", "metrica", "skillcorner")])
            print(f"head {hk:7s} on labels {tn:9s} spellF1 " + " ".join(f"{x:.3f}" for x in per[:, 0]) + f"  mean {per[:, 0].mean():.3f}"
                  f" | alive/dead {per[:, 1].mean():.3f} team {per[:, 2].mean():.3f}")


if __name__ == "__main__":
    main()
