"""What the players-only net of deep/ says about each frame, as columns for the control classifier.

    python external/net_features.py [--tag bt] [match ...]   # -> data/cache/netfeat_<tag>_<match>.npz

For every external match, the fold net that never saw the match's provider
(data/cache/deep/<tag>/<provider>/net.pt) is run over each half, and its outputs are laid out on
the frames external/control_features.py caches: its p(H, A, D), which team its pointer (the "who
is on the ball" stage, taught by the ball on other matches) leans to, and how that pointer and its
guess of the ball's position sit against the ball the decoder is shown. The net reads the players
only; the ball enters here, outside the net, and only as the decoder sees it (--seen 0.59).

Because each match is read by a net that did not train on its provider, a classifier trained on
these columns learns how far to trust a net on a provider it has not seen, which is what BAR-ATH
is to deep/models/bt.pt. columns() is shared with the BAR-ATH path (hmm/decode.py).
"""

import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "external")]
from hmm.net_clue import half_columns, net_outputs  # noqa: E402

CACHE = os.environ.get("CONTROL_CACHE", os.path.join(ROOT, "data", "cache"))


def cache_path(tag, name):
    return os.path.join(CACHE, "netfeat_{}_{}.npz".format(tag, name))


def build(path, tag, nets):
    from control_features import match_ball
    name = os.path.basename(path)[:-4]
    out = cache_path(tag, name)
    if os.path.exists(out):
        return name
    d = np.load(path)
    fold = name.split("_")[0]
    if fold not in nets:
        from net import load_net
        nets[fold] = load_net(os.path.join(CACHE, "deep", tag, fold, "net.pt"))
    ball = match_ball(name, d)
    frames = np.arange(0, len(d["labels"]), 5)
    halves = [(int(a), int(b)) for a, b in d["halves"]]
    N, Ns, names, P = None, None, None, np.full((len(frames), 3), 1 / 3, np.float32)
    for a, b in halves:
        o = net_outputs(nets[fold], d["players"], ball, d["team"], a, b, tuple(d["pitch"]),
                        int(d["fps"]))
        inside = np.flatnonzero((frames >= a) & (frames <= b))
        idx = np.clip(np.round((frames[inside] - a) / 5).astype(int), 0, len(o["p"]) - 1)
        for swap in (False, True):
            cols, names = half_columns(o, swap)
            if N is None:
                N = np.full((len(frames), len(names)), np.nan, np.float32)
                Ns = N.copy()
            (Ns if swap else N)[inside] = cols[idx]
        P[inside] = o["p"][idx]
    np.savez_compressed(out, N=N, Ns=Ns, names=np.array(names), frames=frames, p=P)
    return name


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    sys.path.insert(0, os.path.join(ROOT, "deep"))
    args = sys.argv[1:]
    tag = "bt"
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]
        del args[args.index("--tag"):args.index("--tag") + 2]
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    if args:
        paths = [p for p in paths if any(k in p for k in args)]
    nets = {}
    for p in paths:
        t = time.time()
        print(build(p, tag, nets), "%.0f s" % (time.time() - t), flush=True)
