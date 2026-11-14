"""What the learned model produces at each stage, for the match viewer (viz/combined --deep).

    python deep/explain.py                       # BAR-ATH
    python deep/explain.py --match metrica_1     # an outside match, or 'all'

Writes data/cache/deep/<tag>/viz/<match>.npz with, per half at 5 fps: the H / A / D probability
of every frame, where the ball head puts the ball (pitch metres), and the probability that the
label changes within 0.4 s; and for every video frame the label at each decode stage: the best
guess frame by frame, after the switching cost, and after short runs are absorbed (the answer).

BAR-ATH uses the network trained on all 30 outside matches (deep/models/<tag>.pt). An outside
match uses the fold network that never saw its provider (data/cache/deep/<tag>/<provider>/net.pt),
so what the page shows is held out. Needs torch; the viewer itself then only needs numpy.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), ROOT]
import data as D  # noqa: E402
import decode  # noqa: E402
from net import PossessionNet  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)  # as deep/run.py


def out_path(tag, match):
    return os.path.join(ROOT, "data", "cache", "deep", tag, "viz", (match or "bar_ath") + ".npz")


def heads(net, X, M):
    """All three heads averaged over the four symmetric views: (T, 3) probabilities, (T, 2) ball in
    the canonical frame (home defends the left, units of half a pitch), (T,) change-nearby."""
    net.eval()
    p, ball, edge = 0, 0, 0
    with torch.no_grad():
        for s in (0, 1):
            for m in (0, 1):
                Xv = X
                if s:
                    Xv = D.swap_teams(Xv, np.zeros(len(X), np.int64), np.zeros((len(X), 2), np.float32))[0]
                if m:
                    Xv = D.mirror(Xv, np.zeros((len(X), 2), np.float32))[0]
                cls, b, e = net(torch.from_numpy(Xv)[None], torch.from_numpy(M)[None])[:3]
                q, b = torch.softmax(cls[0], -1).numpy(), b[0].numpy().copy()
                if m:
                    b[:, 1] *= -1
                if s:
                    b, q = -b, q[:, [1, 0, 2]]
                p, ball, edge = p + q / 4, ball + b / 4, edge + torch.sigmoid(e[0]).numpy() / 4
    return p, ball, edge


def explain(match=None, tag="v2"):
    """The dict out_path() holds; computed and saved if it is not there yet."""
    path = out_path(tag, match)
    if os.path.exists(path):
        return dict(np.load(path))
    if match:
        d = np.load(os.path.join(ROOT, "data", "external", match + ".npz"))
        players, ball, team, pitch, fps = d["players"], d["ball"], d["team"], tuple(d["pitch"]), int(d["fps"])
        halves = [tuple(int(v) for v in h) for h in d["halves"]]
        weights = os.path.join(ROOT, "data", "cache", "deep", tag, D.provider(match), "net.pt")
    else:
        import load
        players, ball, team, halves = load.load()
        pitch, fps = load.PITCH, load.FPS
        weights = os.path.join(HERE, "models", tag + ".pt")
    net = PossessionNet(nfeat=D.NFEAT)
    net.load_state_dict(torch.load(weights))
    out = {"halves": np.array(halves), "step": D.STEP,
           # how far each side the convolutions reach, in 5 fps steps (kernel 3: one dilation a side)
           "reach": sum(b.c1.dilation[0] for b in net.tcn)}
    probs = []
    for i, (a, b) in enumerate(halves):
        X, M, _, fr = D.half_tensors(players, ball, team, a, b, pitch, fps)
        p, guess, edge = heads(net, X, M)
        guess = guess * np.array([52.5, 34.0], np.float32)
        if not D.home_defends_left(players, team, a, b):
            guess = -guess
        guess += np.array(pitch, np.float32) / 2
        probs.append(p)
        out.update({f"p{i}": p.astype(np.float32), f"ball{i}": guess.astype(np.float32),
                    f"edge{i}": edge.astype(np.float32),
                    f"err{i}": np.linalg.norm(guess - ball[fr, :2], axis=1).astype(np.float32)})
    n = len(players)
    out["raw"] = decode.labels_from(probs, halves, n, cost=0, min_team=0, min_dead=0)
    out["switch"] = decode.labels_from(probs, halves, n, cost=DECODE["cost"], min_team=0, min_dead=0)
    out["final"] = decode.labels_from(probs, halves, n, **DECODE)
    if not match:  # graded frames before a kickoff are dead, as deep/run.py
        from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
        for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
            for k in ("raw", "switch", "final"):
                out[k][s:kick] = "D"
    out["decode"] = np.array([DECODE["cost"], DECODE["min_team"], DECODE["min_dead"]])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--match", help="an outside match in data/external/, or 'all'; default BAR-ATH")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    matches = [a.match]
    if a.match == "all":
        matches = [None] + [os.path.basename(p)[:-4] for p in D.external_paths()]
    for m in matches:
        explain(m, a.tag)
        print("wrote", out_path(a.tag, m), flush=True)
