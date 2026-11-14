"""The team-naming step on top of a base model, scored held out by provider on folded labels.

    python team/apply.py --base v2 --team t1            # base says live or dead, the team net says who
    python team/apply.py --base bt,bts1 --team t1,t1s1  # seeds averaged on either side

The base keeps its alive/dead probability. Its split of the live share between the teams is
replaced by (or mixed with, --mix) the team net's: P(H) = P_base(live) * q, q the team net's
P(home | live). So the step sits on whichever base is current; nothing in it reads the ball.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import decode  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log1p(-p)


def combine(base, team, mix=1.0):
    """base, team: (T, 3) H/A/D. mix 1 takes the team split from the team net, 0 leaves the base."""
    qb = base[:, 0] / np.clip(base[:, 0] + base[:, 1], 1e-9, None)
    qt = team[:, 0] / np.clip(team[:, 0] + team[:, 1], 1e-9, None)
    q = 1 / (1 + np.exp(-(mix * logit(qt) + (1 - mix) * logit(qb))))
    live = 1 - base[:, 2]
    return np.stack([live * q, live * (1 - q), base[:, 2]], 1)


def mean_probs(tags, name):
    ps = [C.stage1(t, name) for t in tags]
    return [np.mean([p[i] for p in ps], 0) for i in range(len(ps[0]))]


def one(args):
    name, base, team, mixes, decs = args
    raw, fold, spans = C.truth(name)
    pb, pt = mean_probs(base, name), mean_probs(team, name)
    row = []
    for dec in decs:
        sc = lambda probs: [C.spell_f1(g, decode.labels_from(probs, spans, len(raw), **dec)) for g in (fold, raw)]  # noqa: E731
        row.append([sc(pb), sc(pt)] + [sc([combine(b, t, m) for b, t in zip(pb, pt)]) for m in mixes])
    return name, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v2")
    ap.add_argument("--team", default="t1")
    ap.add_argument("--mix", default="1,0.5")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    base, team, mixes = a.base.split(","), a.team.split(","), [float(m) for m in a.mix.split(",")]
    decs = [C.DECODE, dict(C.DECODE, cost=2), dict(C.DECODE, cost=8)]
    names = [os.path.basename(p)[:-4] for p in C.paths()]
    with Pool(a.procs) as pool:
        R = pool.map(one, [(n, base, team, mixes, decs) for n in names])
    S = np.array([r[1] for r in R])  # (matches, decodes, variants, truths)
    labels = [f"base {a.base}", f"team net alone {a.team}"] + [f"base alive/dead + team net, mix {m:g}" for m in mixes]
    for di, dec in enumerate(decs):
        print(f"\ndecode {dec}: spellF1 on folded labels     idsse  metrica  skillc   mean   | raw-label mean | vs base: se, matches up")
        for vi, lab in enumerate(labels):
            f = C.provider_mean(names, S[:, di, vi, 0])
            r = C.provider_mean(names, S[:, di, vi, 1])
            d = S[:, di, vi, 0] - S[:, di, 0, 0]
            print(f"  {lab:42s} " + "  ".join(f"{x:.3f}" for x in f) + f"   | {r[-1]:.3f} | "
                  f"{d.mean():+.3f} (se {d.std(ddof=1) / np.sqrt(len(d)):.3f}, up {int((d > 0).sum())}/{len(d)})")


if __name__ == "__main__":
    main()
