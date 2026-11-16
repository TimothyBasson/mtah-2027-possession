"""Which output to ship, and a stress test of the visibility simulation.

    python alldata/learned/heads.py ps1e            # each head against each label view
    python alldata/learned/heads.py s1 --always-seen  # fold nets re-run with the ball in view through every stoppage

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


def always_seen(tag):
    import torch
    import feats
    import fit as T
    import build
    torch.set_num_threads(2)
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
                bxy[a:b + 1] = feats.bar_like(bxy[a:b + 1], d["labels"][a:b + 1], rng, int(d["fps"]), q=0.0)
                F, _ = feats.half_features(d["players"], bxy, d["team"], a, b, d["pitch"], int(d["fps"]), extra=[("bt", bt[i])])
                cols = [feats.NAMES.index(n) for n in ck["names"]]
                probs.append(T.predict(net, F[:, cols], sym))
            out[name] = probs
            print(name, flush=True)
    return out


def main():
    tag = sys.argv[1]
    res = {}
    if "--always-seen" in sys.argv:
        allp = always_seen(tag)
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(tag), fold, fold + "_*.npz"))):
            name = os.path.basename(f)[:-4]
            raw, fo, spans = S.truths(name)
            z = np.load(f)
            for hk in ("built", "folded", "both"):
                if "--always-seen" in sys.argv:
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
