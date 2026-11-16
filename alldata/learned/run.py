"""Label BAR-ATH with the all-data model and write the submission.

    python alldata/learned/run.py --features          # BAR-ATH's feature cache (before fit.py --fold none)
    python alldata/learned/run.py [--tags psN psNs1] [--out alldata_learned.csv]

Averages the probabilities of the named runs' all-match nets (train.py --fold none), decodes
them as deep/run.py does, and writes evaluation/results/<out>. Reads BAR-ATH's players and
ball, and the players-only net's probabilities (deep/models/bt.pt). No ground truth is read.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE]
sys.path += [os.path.join(ROOT, "deep"), os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), ROOT]
import decode  # noqa: E402
import feats  # noqa: E402
import fit as T  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)


WHICH = "both"


def head(q):
    """The two heads averaged: taught on the labels as built and on the folded labels. Held out,
    the average is within 0.005 of the better head under either labelling (heads.py), and which
    labelling BAR-ATH's truth is closer to is not known."""
    return (q[:, :3] + q[:, 3:]) / 2 if WHICH == "both" else q[:, :3] if WHICH == "built" else q[:, 3:]


def bt_probabilities():
    """The players-only net on BAR-ATH (deep/run.py's probabilities), cached."""
    f = os.path.join(ROOT, "data", "cache", "deep", "bt", "bar_ath_probs.npz")
    if not os.path.exists(f):
        import importlib.util
        spec = importlib.util.spec_from_file_location("deep_run", os.path.join(ROOT, "deep", "run.py"))
        deep_run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deep_run)
        probs, _, _ = deep_run.probabilities("bt")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez_compressed(f, **{f"p{i}": p for i, p in enumerate(probs)})
    z = np.load(f)
    return [z[k] for k in sorted(z.files)]


def features(with_tim=False):
    """Feature rows per half; also cached as data/cache/alldata/bar/BAR-ATH.npz in build.py's
    layout (labels all ungraded) for fit.py --fold none --pseudo."""
    import load
    players, ball, team, halves = load.load()
    bt = bt_probabilities()
    tim_in = [[], []]
    if with_tim:
        import tim
        tim_in = tim.inputs("BAR-ATH", [np.arange(a, b + 1, feats.STEP) for a, b in halves])
    out, save = [], {}
    for i, (a, b) in enumerate(halves):
        F, fr = feats.half_features(players, ball[:, :2], team, a, b, load.PITCH, load.FPS,
                                    extra=[("bt", bt[i])] + tim_in[i])
        out.append(F)
        save.update({f"F0_{i}": F, f"fr_{i}": fr, f"y_{i}": np.full(len(F), -1, np.int8), f"yf_{i}": np.full(len(F), -1, np.int8)})
    names = list(feats.NAMES)
    if with_tim:  # the full set of columns; fit.py drops what a run does not use
        os.makedirs(os.path.join(T.CACHE, "bar"), exist_ok=True)
        np.savez(os.path.join(T.CACHE, "bar", "BAR-ATH.npz"), names=np.array(names), nhalves=len(halves), **save)
    return out, halves, len(players), names


def probabilities(tags, Fs, names):
    probs = [0] * len(Fs)
    for tag in tags:
        path = os.path.join(HERE, "models", tag + ".pt")  # the committed nets; else a fresh fit.py --fold none
        if not os.path.exists(path):
            path = os.path.join(T.CACHE, "runs", tag, "none", "net.pt")
        ck = torch.load(path, map_location="cpu")
        cols = [names.index(n) for n in ck["names"]]
        net = T.Net(len(cols), ck["d"])
        net.load_state_dict(ck["sd"])
        sym = feats.symmetry(ck["names"])
        for i, F in enumerate(Fs):
            probs[i] = probs[i] + head(T.predict(net, F[:, cols], sym)) / len(tags)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["psN", "psNs1"])
    ap.add_argument("--out", default="alldata_learned.csv")
    ap.add_argument("--head", default="both", choices=["both", "built", "folded"],
                    help="both (the file), or one head alone, to compare them on BAR-ATH")
    ap.add_argument("--features", action="store_true", help="only build BAR-ATH's feature cache")
    a = ap.parse_args()
    if a.features:
        features(with_tim=True)
        return
    global WHICH
    WHICH = a.head
    torch.set_num_threads(1)
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    Fs, halves, n, names = features(with_tim=os.path.exists(os.path.join(T.CACHE, "tim", "BAR-ATH.npz")))
    probs = probabilities(a.tags, Fs, names)
    os.makedirs(os.path.join(T.CACHE, "bar_ath"), exist_ok=True)
    np.savez_compressed(os.path.join(T.CACHE, "bar_ath", "+".join(a.tags) + ".npz"), **{f"p{i}": p for i, p in enumerate(probs)})
    pred = decode.labels_from(probs, halves, n, **DECODE)
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
