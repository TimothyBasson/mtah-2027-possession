"""What happens after the classifier: sweeps of how the decoder's path becomes spells, scored
held out against the control-folded labels (the "folded" column of external/score_control.py).

    python external/control_features.py && python external/train_control.py --loso
    python external/post_cache.py           # needs numba as well as scikit-learn
    python external/post_sweep.py [--only rules,shift,snap,bias,hold]

Everything here leaves the classifier, the clue tables and the transitions as v8 has them and
changes one thing after them. Results on 2026-11-15 (v8: 0.644 / 0.816 / 0.480, mean 0.647):

- rules: merging a brief run by its context (external/post_rules.py). Only "td", a possession
  under 1.5-2 s between the other team and a stoppage, helps beyond 0.001: 0.650 / 0.820 / 0.490
  at 2 s, up on 30 of 30 matches; 0.647 / 0.814 / 0.487 with the length chosen on the other two
  providers. hmm.decode.fold_before_stoppage is that rule in one pass (0.649 / 0.820 / 0.490;
  off by default; its comment has the caveat). With POSSESSION_SECONDS at 30 it still adds 0.005.
  Brief stoppages, brief possessions after a stoppage or between two: flat or worse.
- shift: moving every team change, stoppage start or restart by 3-12 frames: within 0.001, worse
  beyond 6 frames. The decoder's changes have no offset to take out.
- snap: moving a team change to the new team's nearest or first touch within 0.5-3 s: 0.000 at
  0.5 s, -0.001 to -0.015 beyond.
- bias: a constant lean towards or away from D: within 0.001.
- hold: interpolating the classifier's output between the frames it is evaluated on (every 5th)
  instead of holding it: +0.0005.
external/post_average.py averages classifiers trained on different providers (no gain) and
external/post_diag.py sizes each kind of error with oracles: the truth's team inside the decoder's
own live stretches would give 0.878, the truth's alive/dead 0.710, and moving every change to a
true one within 1 s only 0.661. What is left is naming the team, not placing the boundary.
"""

import argparse

import numpy as np

import post_lib as pl
import post_rules as pr
import hmm.decode as dec

SECONDS = (0.5, 1, 1.5, 2, 2.5, 3, 4)


def table(ms, label, fn, gt="gt"):
    r = pl.by_set({n: pl.spell(m, fn(m), gt)["mean_f1"] for n, m in ms.items()})
    print("{:34s} {}".format(label, pl.fmt(r)), flush=True)
    return r


def rules(ms):
    grid = {}
    for c, to in (("td", "T"), ("dt", "T"), ("dt", "D"), ("dd", "T"), ("tt", "T"),
                  ("d_same", "T"), ("d_diff", "T")):
        for s in SECONDS:
            label = "{}{} {:g} s".format(c, " -> " + to if c == "dt" else "", s)
            grid[c, to, s] = table(ms, label, lambda m: pr.apply_halves(
                m["path"], m["halves"], {c: s}, int(m["fps"]), to))
    print("td with the length chosen on the other two providers:")
    fair = {}
    for k in pl.SETS:
        others = [o for o in pl.SETS if o != k]
        s = max(SECONDS, key=lambda v: sum(grid["td", "T", v][o] for o in others))
        fair[k] = grid["td", "T", s][k]
        print("  {:12s} {:g} s -> {:.3f}".format(k, s, fair[k]))
    print("  mean {:.4f}".format(np.mean(list(fair.values()))))
    def decoder(m, seconds):  # the decoder's own function: one pass, where the rule above repeats
        out = m["path"].copy()
        for lo, hi in m["halves"]:
            out[lo:hi] = dec.fold_before_stoppage(m["path"][lo:hi],
                                                  int(round(seconds * int(m["fps"]))))
        return out
    for s in (1.5, 2):
        table(ms, "fold_before_stoppage {:g} s".format(s), lambda m: decoder(m, s))
        table(ms, "  labels as they are", lambda m: decoder(m, s), "gt_raw")
    table(ms, "v8, labels as they are", lambda m: m["path"], "gt_raw")


def kind(a, b):
    return "team change" if a < 2 and b < 2 else "to dead" if b == 2 else "from dead"


def shifted(m, which, k):
    p = m["path"]
    out = p.copy()
    for t in np.flatnonzero(p[1:] != p[:-1]) + 1:
        a, b = p[t - 1], p[t]
        if a < 0 or b < 0 or kind(a, b) != which:
            continue
        if k > 0:
            out[t:t + k] = a
        else:
            out[t + k:t] = b
    out[p < 0] = -1
    return out


def shift(ms):
    for which in ("team change", "to dead", "from dead"):
        for k in (-12, -6, -3, 3, 6, 12):
            table(ms, "{} {:+d} frames".format(which, k), lambda m: shifted(m, which, k))


def snapped(m, seconds, first):
    """Each team change moved to the new team's touch within seconds of it: the nearest, or (first)
    the earliest."""
    p, w = m["path"], seconds * int(m["fps"])
    out = p.copy()
    frames, teams = m["touch_frame"], m["touch_team"]
    edges = np.flatnonzero(p[1:] != p[:-1]) + 1
    for t, lo, hi in zip(edges, np.r_[0, edges[:-1]], np.r_[edges[1:], len(p)]):
        a, b = p[t - 1], p[t]
        if a < 0 or b < 0 or a == 2 or b == 2:
            continue
        own = frames[teams == b]
        own = own[(np.abs(own - t) <= w) & (own < hi - 1) & (own > lo + 1)]
        if not len(own):
            continue
        c = own[0] if first else own[np.argmin(np.abs(own - t))]
        if c > t:
            out[t:c] = a
        else:
            out[c:t] = b
    return out


def snap(ms):
    for first in (False, True):
        for s in (0.5, 1, 2, 3):
            table(ms, "{} touch within {:g} s".format("first" if first else "nearest", s),
                  lambda m: snapped(m, s, first))


def bias(ms):
    for b in (-0.1, -0.05, -0.02, 0.02, 0.05, 0.1):
        def lean(m):
            e = pl.emissions(m)
            e[:, 2] += b
            return pl.decode(m, e)
        table(ms, "D {:+.2f} a frame".format(b), lean)


def hold(ms):
    def interpolated(m, stride=5):
        p = np.exp(m["ratio_control_loso"].astype(float))  # the posterior over the prior
        at, x = np.arange(0, len(p), stride), np.arange(len(p))
        p = np.stack([np.interp(x, at, p[at, c]) for c in range(3)], 1)
        return pl.decode(m, m["emit"].astype(float) + dec.CONTROL_WEIGHT * np.log(p))
    table(ms, "posterior interpolated", interpolated)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="rules,shift,snap,bias,hold")
    a = ap.parse_args()
    ms = pl.load_all()
    for m in ms.values():
        m["path"] = pl.decode(m)
    table(ms, "v8", lambda m: m["path"])
    for name in a.only.split(","):
        print("\n" + name)
        globals()[name](ms)


if __name__ == "__main__":
    main()
