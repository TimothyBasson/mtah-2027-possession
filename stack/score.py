"""The learned players-only model (deep/) and the hand-built one (possession/) through one decode,
scored held out by provider. Both read the players only; neither reads the ball.

    python stack/score.py [--tag v1o] [--workers 2] [--knock]

For each provider X in turn (the folds of deep/train.py and external/heldout.py):
  - deep/'s probabilities for X's matches come from the net that never saw X (kind "test"), and
    for that fold's validation matches (other providers, kept out of training; kind "val")
  - possession/'s two logistic models are fitted on the other two providers minus those
    validation matches
  - the stack's weights and every decode setting are picked on the validation matches only
  - X's matches are scored
The mean counts each provider equally. --knock removes players as BAR-ATH's tracker loses them
(external/score.py drop_players) from possession/'s input; deep/'s saved probabilities are not
recomputed, so use it only with a tag whose probabilities were predicted on knocked-out tracking.

Variants:
  deep        deep/'s log-probabilities alone
  poss        possession/'s alone
  blend w     deep's log-probabilities + w x possession's
  stack       two logistics fitted on the validation matches: home against away on (deep's log
              H/A ratio, possession's log-odds of home), dead against live on (deep's log D/live
              ratio, possession's log-odds of dead)
All share one decode: a three-state chain with a flat switching cost, then short runs absorbed
(deep/decode.py's), over the same grid.
"""

import argparse
import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), os.path.join(ROOT, "deep"),
                os.path.join(ROOT, "evaluation"), ROOT]

PROVIDERS = ("idsse", "metrica", "skillcorner")
VAL = {"idsse": 1, "metrica": 1, "skillcorner": 3}  # as deep/train.py
METRICS = ("spellF1", "macroF1", "alive/dead", "team")
GRID = [dict(cost=c, min_team=t, min_dead=d) for c in (2, 4, 8, 16) for t, d in ((1, 1), (2, 3))]
BLEND = (0.25, 0.5, 1.0)
VARIANTS = ["deep", "poss"] + [f"blend {w:g}" for w in BLEND] + ["stack"]


def provider(name):
    return os.path.basename(name).split("_")[0]


def ext(name):
    return os.path.join(ROOT, "data", "external", name + ".npz")


def split(fold):
    """(train names, val names, test names) as deep/train.py splits them."""
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(ext("*")))
    train, val, test = [], [], []
    for prov in PROVIDERS:
        ms = [n for n in names if provider(n) == prov]
        if prov == fold:
            test += ms
        else:
            train += ms[:-VAL[prov]]
            val += ms[-VAL[prov]:]
    return train, val, test


def point_model_at(d):
    import features
    import model
    halves = [tuple(int(v) for v in h) for h in d["halves"]]
    features.PITCH = model.PITCH = tuple(d["pitch"])
    model.HALVES = halves
    mask = np.zeros(len(d["labels"]), bool)
    for a, b in halves:
        mask[a:b + 1] = True
    model.graded_mask = lambda n: mask
    return halves


def fit_possession(args):
    fold, knock = args
    import model
    from score import drop_players
    ms = []
    for n in split(fold)[0]:
        d = np.load(ext(n))
        halves = [tuple(int(v) for v in h) for h in d["halves"]]
        players = drop_players(d["players"], halves, model.FPS, n + ".npz") if knock else d["players"]
        ms.append((provider(n), players, d["labels"], halves, d["team"], d["pitch"]))
    return model.fit(ms)


def observe(args):
    """One match in one fold: both models' per-frame log-odds at 25 fps, per half."""
    fold, name, kind, fit, tag, knock = args
    import model
    from decode import upsample
    from score import drop_players
    d = np.load(ext(name))
    halves = point_model_at(d)
    players = drop_players(d["players"], halves, model.FPS, name + ".npz") if knock else d["players"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = model.explain(dict(players=players, team=d["team"]), fit | {"decode": "absorb"})
    z = np.load(os.path.join(ROOT, "data", "cache", "deep", tag, fold, name + ".npz"))
    assert str(z["kind"]) == kind, (fold, name, kind)
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))  # noqa: E731
    out = []
    for i, (a, b) in enumerate(halves):
        dp = np.log(np.clip(upsample(z[f"p{i}"], b - a + 1), 1e-6, 1))
        out.append(dict(deep=dp, home=lo(e["p_home"][a:b + 1]), dead=lo(e["p_dead"][a:b + 1])))
    return dict(fold=fold, name=name, kind=kind, halves=halves, labels=d["labels"], obs=out)


def poss_logp(h):
    ls = lambda x: -np.logaddexp(0, -x)  # noqa: E731  log sigmoid
    return np.column_stack([ls(-h["dead"]) + ls(h["home"]), ls(-h["dead"]) + ls(-h["home"]), ls(h["dead"])])


def stack_x(h):
    dp = h["deep"]
    return (np.column_stack([dp[:, 0] - dp[:, 1], h["home"]]),
            np.column_stack([dp[:, 2] - np.logaddexp(dp[:, 0], dp[:, 1]), h["dead"]]))


def fit_stack(ms):
    """The two stacking logistics from a fold's validation matches, every 5th labelled frame."""
    import model
    team, dead, provs = ([], []), ([], []), []
    for m in ms:
        tx, dx, lab = [], [], []
        for h, (a, b) in zip(m["obs"], m["halves"]):
            t, dd = stack_x(h)
            tx.append(t[::5]); dx.append(dd[::5]); lab.append(m["labels"][a:b + 1:5])
        tx, dx, lab = np.vstack(tx), np.vstack(dx), np.concatenate(lab)
        g = np.isin(lab, ["H", "A", "D"])
        dead[0].append(dx[g]); dead[1].append(lab[g] == "D")
        g = np.isin(lab, ["H", "A"])
        team[0].append(tx[g]); team[1].append(lab[g] == "H")
        provs.append(provider(m["name"]))
    return dict(team=model.fit_logistic(*team, provs), dead=model.fit_logistic(*dead, provs))


def emissions(h, variant, stack):
    if variant == "deep":
        return h["deep"]
    if variant == "poss":
        return poss_logp(h)
    if variant.startswith("blend"):
        return h["deep"] + float(variant.split()[1]) * poss_logp(h)
    import model
    t, dd = stack_x(h)
    return poss_logp(dict(home=model.logit(t, stack["team"]), dead=model.logit(dd, stack["dead"])))


def viterbi(logp, cost):
    """deep/decode.py's chain (a flat cost for any change of state), in plain Python floats: the
    same path, several times faster than a NumPy step per frame."""
    rows = logp.tolist()
    v0, v1, v2 = rows[0]
    back = []
    for e0, e1, e2 in rows[1:]:
        if v0 >= v1 and v0 >= v2:
            best, top = 0, v0
        elif v1 >= v2:
            best, top = 1, v1
        else:
            best, top = 2, v2
        jump = top - cost
        b0, b1, b2 = 0, 1, 2
        if v0 < jump:
            v0, b0 = jump, best
        if v1 < jump:
            v1, b1 = jump, best
        if v2 < jump:
            v2, b2 = jump, best
        back.append((b0, b1, b2))
        v0, v1, v2 = v0 + e0, v1 + e1, v2 + e2
    state = 0 if v0 >= v1 and v0 >= v2 else (1 if v1 >= v2 else 2)
    path = [state]
    for b in reversed(back):
        state = b[state]
        path.append(state)
    return np.array(path[::-1], np.int8)


def score_match(args):
    """(len(VARIANTS), len(GRID), 4) scores of one match in one fold."""
    m, stack = args
    import possession_metrics as pm
    from decode import LETTERS, absorb
    truth = pm.label_codes(m["labels"])
    out = np.zeros((len(VARIANTS), len(GRID), 4))
    for vi, variant in enumerate(VARIANTS):
        E = [emissions(h, variant, stack) for h in m["obs"]]
        paths = {}
        for gi, g in enumerate(GRID):
            pred = np.full(len(truth), "X", "<U1")
            for hi, (e, (a, b)) in enumerate(zip(E, m["halves"])):
                if (hi, g["cost"]) not in paths:
                    paths[hi, g["cost"]] = viterbi(e, g["cost"])
                p = absorb(paths[hi, g["cost"]], {0: g["min_team"] * 25, 1: g["min_team"] * 25, 2: g["min_dead"] * 25})
                pred[a:b + 1] = LETTERS[p]
            r = pm.evaluate(truth, pm.label_codes(pred), bootstrap_n=0)
            out[vi, gi] = r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1o")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--knock", action="store_true")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        fits = dict(zip(PROVIDERS, ex.map(fit_possession, [(f, a.knock) for f in PROVIDERS])))
        jobs = [(f, n, kind, fits[f], a.tag, a.knock) for f in PROVIDERS
                for kind, ns in zip(("val", "test"), split(f)[1:]) for n in ns]
        ms = list(ex.map(observe, jobs))
        stacks = {f: fit_stack([m for m in ms if m["fold"] == f and m["kind"] == "val"]) for f in PROVIDERS}
        scores = list(ex.map(score_match, [(m, stacks[m["fold"]]) for m in ms]))
    for f in PROVIDERS:
        print(f"stack fitted on {f}'s validation matches: team {stacks[f]['team']['w']}, dead {stacks[f]['dead']['w']}")
    print(f"\n{'variant':10s} {'held out':12s} " + "  ".join(f"{x:>10s}" for x in METRICS) + "   best here   picked on validation")
    for vi, variant in enumerate(VARIANTS):
        rows = []
        for f in PROVIDERS:
            val = [(m, s[vi]) for m, s in zip(ms, scores) if m["fold"] == f and m["kind"] == "val"]
            provs = sorted({provider(m["name"]) for m, _ in val})  # each validation provider counts equally
            v = np.mean([np.mean([s for m, s in val if provider(m["name"]) == p], 0) for p in provs], 0)
            pick = int(v[:, 0].argmax())
            t = np.mean([s[vi] for m, s in zip(ms, scores) if m["fold"] == f and m["kind"] == "test"], 0)
            rows.append(t[pick])
            print(f"{variant:10s} {f:12s} " + "  ".join(f"{x:10.3f}" for x in t[pick]) + f"   {t[:, 0].max():9.3f}   {GRID[pick]}")
        print(f"{variant:10s} {'mean':12s} " + "  ".join(f"{x:10.3f}" for x in np.mean(rows, 0)) + "\n")


if __name__ == "__main__":
    main()
