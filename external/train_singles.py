"""Held-out posteriors from control classifiers trained on ONE provider each, for averaging with
the pooled held-out model (external/post_sweep.py).

    python external/train_singles.py      # -> data/cache/control_single_<provider>.npz

Each file holds, for every match of the other two providers, the posteriors of the model trained
on <provider> alone, in the layout of train_control.py --loso. Same features, same settings.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external"))
import train_control as tc  # noqa: E402
from control_features import CACHE  # noqa: E402
from hmm.control import hold  # noqa: E402


def main():
    matches = tc.load_matches()
    for k in tc.SETS:
        X, y, w = tc.rows(matches, [n for n in matches if n.startswith(k)])
        t = time.time()
        model = tc.fit(X, y, w)
        prior = tc.class_shares(y, w)
        out = {}
        for n in (n for n in matches if not n.startswith(k)):
            x, xs, labels, _, frames, n_frames = matches[n]
            out[n] = hold(tc.posteriors(model, x, xs).astype(np.float32), n_frames)
            out["prior_" + n] = prior
        path = os.path.join(CACHE, "control_single_{}.npz".format(k))
        np.savez_compressed(path, **out)
        print("{} alone: {} rows, {:.0f} s -> {}".format(k, len(y), time.time() - t, path),
              flush=True)


if __name__ == "__main__":
    main()
