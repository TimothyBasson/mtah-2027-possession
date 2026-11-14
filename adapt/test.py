"""Score a tag's fold nets on held-out matches whose players are lost as BAR-ATH loses them.

    python adapt/test.py --tag v2 --cond full bar old bar+fill15

Conditions (what is done to the held-out match's tracking before the saved fold net reads it):
    full        nothing
    old         external/score.py drop_players: flat-rate losses and bursts dropped anywhere
    bar         adapt/dropout.py simulate: BAR-ATH's measured losses by situation
    bar+fillN   bar, then adapt/impute.py fill with gaps up to N seconds joined
    (old+fillN and full+fillN work too)

Probabilities go to data/cache/deep/<tag>@<cond>/<fold>/<match>.npz (held-out matches only).
Decode is fixed, not picked: cost 4 with 1 s / 1 s runs against the labels as given, and 2 s / 3 s
against the labels folded to controlled possession (external/control.py), as the evaluation
review recommends. Mean over providers, each counting equally. Nothing is trained here, and any
base model's fold nets in the same layout can be scored the same way.

Also printed: on frames that are truly live (H or A) and have fewer than 20 players tracked, how
often the model calls them dead. BAR-ATH's public 5 minutes have no such frames, so this is the
only place that question can be asked.
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"),
                os.path.join(ROOT, "possession"), os.path.join(ROOT, "evaluation"), ROOT]

RAW = dict(cost=4, min_team=1, min_dead=1)
FOLDED = dict(cost=4, min_team=2, min_dead=3)


def apply(cond, d, name):
    """(players, team) for one condition."""
    import dropout
    import impute
    players, team = d["players"], d["team"]
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    fps = int(d["fps"])
    for part in cond.split("+"):
        if part == "old":
            from score import drop_players
            players = drop_players(players, halves, fps, name)
        elif part == "bar":
            players = dropout.simulate(players, halves, fps, name)
        elif part.startswith("fill"):
            players, team = impute.fill(players, team, halves, fps, float(part[4:]))
        elif part != "full":
            raise SystemExit("unknown condition " + part)
    return players, team


def folded_labels(name, d):
    path = os.path.join(ROOT, "data", "cache", "control_labels", name + ".npy")
    if not os.path.exists(path):
        from control import control_labels
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, control_labels(d))
    return np.load(path)


def one(args):
    tag, cond, fold, name = args
    import torch
    torch.set_num_threads(1)
    import data as D
    import decode
    import possession_metrics as pm
    from net import PossessionNet
    from train import predict_half
    base = os.path.join(ROOT, "data", "cache", "deep")
    d = np.load(os.path.join(ROOT, "data", "external", name + ".npz"))
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    out = os.path.join(base, f"{tag}@{cond}", fold, name + ".npz")
    players, team = apply(cond, d, name + ".npz")
    if cond == "full" and not os.path.exists(out):
        out = os.path.join(base, tag, fold, name + ".npz")
    if not os.path.exists(out):
        net = PossessionNet(nfeat=D.NFEAT)
        net.load_state_dict(torch.load(os.path.join(base, tag, fold, "net.pt")))
        probs = {}
        for i, (s, e) in enumerate(halves):
            X, M, bxy, _ = D.half_tensors(players, d["ball"], team, s, e, tuple(d["pitch"]), int(d["fps"]))
            probs[f"p{i}"] = predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, kind="test", **probs)
    z = np.load(out)
    probs = [z[f"p{i}"] for i in range(len(halves))]
    n = len(d["labels"])
    res = {}
    for key, truth, dec in (("raw", d["labels"], RAW), ("folded", folded_labels(name, d), FOLDED)):
        pred = decode.labels_from(probs, halves, n, **dec)
        r = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)
        res[key] = (r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"])
        if key == "raw":
            tracked = np.isfinite(players[..., 0]).sum(1)
            live = np.isin(truth, ["H", "A"])
            few = (tracked < 20) & (tracked > 0)
            res["few"] = (int((live & few).sum()), int((live & few & (pred == "D")).sum()),
                          int((few & (truth != "X")).sum()), int((live & ~few).sum()),
                          int((live & ~few & (pred == "D")).sum()))
    return fold, name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--cond", nargs="+", default=["full", "old", "bar"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    base = os.path.join(ROOT, "data", "cache", "deep", a.tag)
    jobs = []
    for fold in sorted(os.listdir(base)):
        if not os.path.exists(os.path.join(base, fold, "net.pt")):
            continue
        for f in sorted(glob.glob(os.path.join(base, fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                jobs.append((fold, os.path.basename(f)[:-4]))
    print(f"{'condition':12s} labels   IDSSE  Metrica  SkillC   mean  | macroF1 alive/dead team | "
          "live frames with <20 tracked: share of graded, called dead (other live frames)")
    with ProcessPoolExecutor(a.workers) as ex:
        for cond in a.cond:
            res = list(ex.map(one, [(a.tag, cond, fold, name) for fold, name in jobs]))
            folds = sorted({f for f, _, _ in res})
            few = np.sum([r["few"] for _, _, r in res], 0)
            for key in ("raw", "folded"):
                per = np.array([np.mean([r[key] for f, _, r in res if f == fold], 0) for fold in folds])
                m = per.mean(0)
                tail = ""
                if key == "raw":
                    tail = (f" | {few[2] and few[0] / max(few[2], 1):.2f} of the <20 frames are live; "
                            f"called dead {few[1] / max(few[0], 1):.2f} ({few[4] / max(few[3], 1):.2f})")
                print(f"{cond:12s} {key:7s} " + "  ".join(f"{x:.3f}" for x in per[:, 0]) + f"  {m[0]:.3f}  | "
                      + "  ".join(f"{x:.3f}" for x in m[1:]) + tail, flush=True)


if __name__ == "__main__":
    main()
