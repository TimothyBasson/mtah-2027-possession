"""Label BAR-ATH with the learned model and write the submission.

    python deep/run.py [--tag sw]      # -> evaluation/results/deep.csv (--tag bt, --tag v2: earlier models)

Uses the network trained on all 30 outside matches (deep/train.py --fold none). Players only.
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
from net import load_net  # noqa: E402
from train import predict_half  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)  # what two of the three held-out folds picked


def probabilities(tag):
    import load
    players, ball, team, halves = load.load()
    net = load_net(os.path.join(HERE, "models", tag + ".pt"), D.NFEAT)  # train.py --fold none
    probs = []
    for a, b in halves:
        X, M, bxy, fr = D.half_tensors(players, ball, team, a, b, load.PITCH, load.FPS)
        probs.append(predict_half(net, dict(X=X, M=M, y=np.zeros(len(X), np.int64), ball=bxy)))
    return probs, halves, len(players)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sw")
    ap.add_argument("--out", default="deep.csv")
    ap.add_argument("--tags", action="store_true",
                    help="use Smart Tagging set pieces to draw stoppages the model plays through (stops/tags.py)")
    ap.add_argument("--threads", type=int, default=1,
                    help="pinned so a rerun does the same arithmetic; 1, 4 and 8 give the same file (README)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    probs, halves, n = probabilities(a.tag)
    os.makedirs(os.path.join(ROOT, "data", "cache", "deep", a.tag), exist_ok=True)
    import load
    np.savez_compressed(os.path.join(ROOT, "data", "cache", "deep", a.tag, f"bar_ath_probs{load.cache_suffix()}.npz"),
                        **{f"p{i}": p for i, p in enumerate(probs)})
    pred = decode.labels_from(probs, halves, n, **DECODE)
    if a.tags:  # not players-only: Metrica's automatic set-piece tags, where the model shows no stoppage
        sys.path.insert(0, os.path.join(ROOT, "stops"))
        import tags
        starts, mins = tags.set_pieces(), {k: 25 * v for k, v in ((0, DECODE["min_team"]), (1, DECODE["min_team"]), (2, DECODE["min_dead"]))}
        for p, (s, e) in zip(probs, halves):
            lp = np.log(np.clip(decode.upsample(p, e - s + 1), 1e-6, 1))
            lp, hit = tags.apply(lp, decode.absorb(decode.viterbi(lp, DECODE["cost"]), mins), s, starts)
            if hit:
                pred[s:e + 1] = decode.LETTERS[decode.absorb(decode.viterbi(lp, DECODE["cost"]), mins)]
                print("stoppage drawn for the set pieces tagged at frames", hit)
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):  # graded frames before a kickoff are D
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
