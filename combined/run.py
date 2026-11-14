"""Label BAR-ATH with hmm v7 plus deep/ as one more clue where there is no usable ball.

    python combined/run.py [--tag v2] [--seen 0] [--unseen 0.2]   # -> evaluation/results/hmm_v7_deep.csv

hmm/decode.py runs exactly as it does for hmm_v7.csv; the only difference is one more term added
to its direct (control) clue: a weight times (log of deep/'s probability of the state minus the
log of the class share), the form combined/deep_in_hmm.py scores on the outside matches. Nothing
under hmm/ or deep/ changes. An experiment, not the submission.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "possession"), os.path.join(ROOT, "external")]
import hmm.decode as dec  # noqa: E402
from hmm import trees  # noqa: E402
from hmm.clues import ball_is_usable  # noqa: E402


def deep_term(tag, seen, unseen, threads=1):
    """(video frames, 3): weight x (log deep probability - log class share), 0 outside the halves."""
    import torch
    import decode
    import load
    import run as deep_run
    torch.set_num_threads(threads)
    probs, halves, n = deep_run.probabilities(tag)
    ball = load.load()[1]
    usable = ball_is_usable(ball[:, 0].astype(float), ball[:, 1].astype(float), load.PITCH)
    prior = np.log(trees.load(dec.CONTROL_MODEL)["prior"])
    out = np.zeros((n, 3))
    for p, (a, b) in zip(probs, halves):
        out[a:b + 1] = np.log(np.clip(decode.upsample(p, b - a + 1), 1e-6, 1)) - prior
    return out * np.where(usable, seen, unseen)[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--seen", type=float, default=0.0, help="weight where the ball is usable")
    ap.add_argument("--unseen", type=float, default=0.2, help="weight where it is not")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v7_deep.csv"))
    a = ap.parse_args()
    term = deep_term(a.tag, a.seen, a.unseen)
    control = dec.load_control

    def with_deep(**kw):  # main() multiplies what this returns by CONTROL_WEIGHT
        c = control(**kw)
        assert c.shape == term.shape, (c.shape, term.shape)
        return c + term / dec.CONTROL_WEIGHT

    dec.load_control = with_deep
    sys.argv = [sys.argv[0], "--out", a.out]
    dec.main()


if __name__ == "__main__":
    main()
