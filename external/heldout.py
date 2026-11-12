"""Held-out-provider scoring: pick settings on two providers, score them on the third.

    uv run python external/heldout.py [--model DIR] [param=v1,v2,... ...]

Each comma list is a grid; every combination is scored on every outside match. Then, for each
provider (idsse, metrica, skillcorner) in turn, the combination with the best mean spellF1 on the
OTHER providers is scored on that one. The held-out mean weights the three providers equally, so
SkillCorner's 20 matches don't drown out Metrica's 3.

A model with a fit(matches) function (possession/model.py) is refitted for each held-out provider
on the other two, and its fitted settings are passed in with every grid combination, so nothing
fitted ever sees the provider being scored.

This is the number to quote for a change: it says whether an idea helps on a provider whose
labelling it never saw, not whether it fits these matches. With no grid, it scores the model's
own defaults per provider, which are then all "held out" only if they were set by reasoning.
"""

import argparse
import glob
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from score import drop_ball, frames, pm  # noqa: E402

METRICS = ("spellF1", "macroF1", "alive/dead", "team")


def provider(path):
    return os.path.basename(path).split("_")[0]


def score_match(args):
    """Scores of one match under every combination: {combo index: (spellF1, macroF1, alive/dead, team)}."""
    path, model_dir, combos = args
    sys.path.insert(0, model_dir)
    import features
    import model
    d = np.load(path)
    halves = [tuple(h) for h in d["halves"]]
    mask = np.zeros(len(d["labels"]), bool)
    for s, e in halves:
        mask[s:e + 1] = True
    features.PITCH = model.PITCH = tuple(d["pitch"])
    model.HALVES = halves
    model.graded_mask = lambda n: mask
    F = frames(d["players"], drop_ball(d["ball"], 1.0, model.FPS, labels=d["labels"]), d["team"], features)
    truth = pm.label_codes(d["labels"])
    out = []
    for params in combos:
        r = pm.evaluate(truth, pm.label_codes(model.predict(F, params)), bootstrap_n=0)
        out.append((r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]))
    return out


def parse_grid(items):
    keys, values = [], []
    for item in items:
        k, v = item.split("=")
        keys.append(k)
        values.append([x if x.isalpha() else float(x) for x in v.split(",")])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def fit_fold(args):
    model_dir, train = args
    sys.path.insert(0, model_dir)
    import model
    ms = []
    for path in train:
        d = np.load(path)
        ms.append((provider(path), d["players"], d["labels"], [tuple(h) for h in d["halves"]], d["team"]))
    return model.fit(ms)


def heldout(paths, scores, combos, only=None):
    """Rows (provider, chosen combo, its scores there, and the best in-sample combo's spellF1 there).
    `only` limits the rows to one held-out provider (its scores come from that fold's refit)."""
    sets = sorted({provider(p) for p in paths})
    rows = []
    for test in sets:
        if only and test != only:
            continue
        test_i = [i for i, p in enumerate(paths) if provider(p) == test]
        train_sets = [s for s in sets if s != test]
        # mean over the training providers, each provider weighted equally
        train = np.mean([scores[[i for i, p in enumerate(paths) if provider(p) == s]].mean(0) for s in train_sets], 0)
        pick = int(np.argmax(train[:, 0]))
        here = scores[test_i].mean(0)
        rows.append((test, combos[pick], here[pick], here[:, 0].max()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "possession"))
    ap.add_argument("grid", nargs="*")
    a = ap.parse_args()
    combos = parse_grid(a.grid)
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    model_dir = os.path.abspath(a.model)
    sys.path.insert(0, model_dir)
    import model
    sets = sorted({provider(p) for p in paths})
    with ProcessPoolExecutor() as ex:
        if hasattr(model, "fit"):  # one refit per held-out provider, on the other two
            fits = list(ex.map(fit_fold, [(model_dir, [p for p in paths if provider(p) != s]) for s in sets]))
            rows = []
            for s, fitted in zip(sets, fits):
                cs = [c | fitted for c in combos]
                scores = np.array(list(ex.map(score_match, [(p, model_dir, cs) for p in paths])))
                rows += [(t, {k: v for k, v in c.items() if k not in fitted}, x, b)
                         for t, c, x, b in heldout(paths, scores, cs, only=s)]
        else:
            scores = np.array(list(ex.map(score_match, [(p, model_dir, combos) for p in paths])))
            rows = heldout(paths, scores, combos)
    # scores: (match, combo, metric)
    print(f"{len(combos)} setting(s) x {len(paths)} matches; settings picked on the other providers"
          + ("; fitted weights refitted without the held-out provider" if hasattr(model, "fit") else ""))
    print(f"{'held out':12s} " + "  ".join(f"{m:>10s}" for m in METRICS) + "   best here   picked")
    for test, combo, s, best in rows:
        print(f"{test:12s} " + "  ".join(f"{x:10.3f}" for x in s) + f"   {best:9.3f}   {combo or 'defaults'}")
    m = np.mean([s for _, _, s, _ in rows], 0)
    print(f"{'mean':12s} " + "  ".join(f"{x:10.3f}" for x in m))


if __name__ == "__main__":
    main()
