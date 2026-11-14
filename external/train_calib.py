"""Cross-provider posteriors for fitting a temperature without the held-out set's labels.

    python external/control_features.py
    python external/train_calib.py                  # -> data/cache/control_inner.npz

external/score_calib.py tests label-free corrections of the control classifier per target match.
One of them needs a temperature, and it may not be fitted on the provider being scored. So for the
provider left out, the temperature is fitted on the two training providers, where each is scored
by a classifier trained on the other one alone: the nearest thing to "a provider the classifier
never saw" that the training side can offer. This script trains the three single-provider
classifiers and saves, for every match of the other two providers, the posteriors on every STEP-th
graded frame together with the folded labels.

Keys: "<train>__<match>" posteriors (n, 3), "y__<match>" label codes (n,).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
import train_control as tc  # noqa: E402
from control_features import CACHE  # noqa: E402


def main():
    matches = tc.load_matches()
    out = {}
    for a in tc.SETS:
        t = time.time()
        X, y = tc.rows(matches, [n for n in matches if n.startswith(a)])
        model = tc.fit(X, y)
        for n in (n for n in matches if not n.startswith(a)):
            xa, xb, labels, _ = matches[n]
            g = np.flatnonzero(labels >= 0)[::tc.STEP]
            out["{}__{}".format(a, n)] = tc.posteriors(model, xa[g], xb[g]).astype(np.float32)
            out["y__" + n] = labels[g].astype(np.int8)
        print("{:12s} alone: {} rows, {} iterations, {:.0f} s".format(
            a, len(y), model.n_iter_, time.time() - t), flush=True)
    path = os.path.join(CACHE, "control_inner.npz")
    np.savez_compressed(path, **out)
    print("->", path)


if __name__ == "__main__":
    main()
