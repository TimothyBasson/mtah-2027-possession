"""Does averaging control classifiers trained on different providers beat the pooled one?

    python external/train_control.py --loso && python external/train_singles.py
    python external/post_cache.py && python external/post_average.py

For each held-out provider there are three classifiers that never saw it: the pooled one (trained
on the other two providers, what v8 scores with) and one trained on each of the other two alone.
Their posteriors are combined before the decoder reads them; everything else is v8.
"""

import os

import numpy as np

import post_lib as pl
import hmm.decode as dec
from control_features import CACHE


def main():
    ms = pl.load_all()
    pooled = dict(np.load(os.path.join(CACHE, "control_loso.npz")))
    single = {k: dict(np.load(os.path.join(CACHE, "control_single_{}.npz".format(k))))
              for k in pl.SETS}

    def parts(n):
        others = [k for k in pl.SETS if not n.startswith(k)]
        return [(pooled[n], pooled["prior_" + n])] + [(single[k][n], single[k]["prior_" + n])
                                                     for k in others]

    def ratio(p, prior):
        return np.log(np.clip(p, 1e-6, 1)) - np.log(prior)

    def geometric(n, w):
        return sum(wi * ratio(p, pr) for wi, (p, pr) in zip(w, parts(n)))

    def arithmetic(n, w):
        ps = parts(n)
        return ratio(sum(wi * p for wi, (p, _) in zip(w, ps)),
                     sum(wi * pr for wi, (_, pr) in zip(w, ps)))

    variants = [("pooled (v8)", geometric, (1, 0, 0)),
                ("two singles, geometric", geometric, (0, .5, .5)),
                ("two singles, arithmetic", arithmetic, (0, .5, .5)),
                ("all three, geometric", geometric, (1 / 3, 1 / 3, 1 / 3)),
                ("all three, arithmetic", arithmetic, (1 / 3, 1 / 3, 1 / 3)),
                ("pooled 1/2 + singles, geometric", geometric, (.5, .25, .25)),
                ("pooled 1/2 + singles, arithmetic", arithmetic, (.5, .25, .25)),
                ("pooled 3/4 + singles, arithmetic", arithmetic, (.75, .125, .125))]
    for label, fn, w in variants:
        f1 = {n: pl.spell(m, pl.decode(m, m["emit"].astype(float)
                                       + dec.CONTROL_WEIGHT * fn(n, w)))["mean_f1"]
              for n, m in ms.items()}
        print("{:34s} {}".format(label, pl.fmt(pl.by_set(f1))), flush=True)


if __name__ == "__main__":
    main()
