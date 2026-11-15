"""Label BAR-ATH with the all-data model and write the submission.

    python alldata/learned/run.py --tags f1 f1s1 [--out alldata_learned.csv]

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
sys.path[:0] = [HERE, os.path.join(ROOT, "deep"), os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), ROOT]
import decode  # noqa: E402
import feats  # noqa: E402
import train as T  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)


def bt_probabilities():
    """The players-only net on BAR-ATH (deep/run.py's probabilities), cached."""
    f = os.path.join(ROOT, "data", "cache", "deep", "bt", "bar_ath_probs.npz")
    if not os.path.exists(f):
        import run as deep_run
        probs, _, _ = deep_run.probabilities("bt")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez_compressed(f, **{f"p{i}": p for i, p in enumerate(probs)})
    z = np.load(f)
    return [z[k] for k in sorted(z.files)]


def features(extra_fn=None):
    import load
    players, ball, team, halves = load.load()
    bt = bt_probabilities()
    out = []
    for i, (a, b) in enumerate(halves):
        extra = [("bt", bt[i])] + (extra_fn(i, a, b) if extra_fn else [])
        F, fr = feats.half_features(players, ball[:, :2], team, a, b, load.PITCH, load.FPS, extra=extra)
        out.append(F)
    return out, halves, len(players), list(feats.NAMES)


def probabilities(tags, Fs, names):
    probs = [0] * len(Fs)
    for tag in tags:
        ck = torch.load(os.path.join(T.CACHE, "runs", tag, "none", "net.pt"), map_location="cpu")
        cols = [names.index(n) for n in ck["names"]]
        net = T.Net(len(cols), ck["d"])
        net.load_state_dict(ck["sd"])
        sym = feats.symmetry(ck["names"])
        for i, F in enumerate(Fs):
            probs[i] = probs[i] + T.predict(net, F[:, cols], sym) / len(tags)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="alldata_learned.csv")
    a = ap.parse_args()
    torch.set_num_threads(1)
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    Fs, halves, n, names = features()
    probs = probabilities(a.tags, Fs, names)
    os.makedirs(os.path.join(T.CACHE, "bar_ath"), exist_ok=True)
    np.savez_compressed(os.path.join(T.CACHE, "bar_ath", "+".join(a.tags) + ".npz"), **{f"p{i}": p for i, p in enumerate(probs)})
    pred = decode.labels_from(probs, halves, n, **DECODE)
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
