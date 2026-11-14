"""A first combination of the players-only model (possession/) with the ball-based decoder (hmm/),
scored held out by provider. An experiment: nothing under possession/, hmm/ or seq/ changes.

    uv run python combined/score.py [--seen 0.59] [--workers 5]
    uv run python combined/score.py --bar-ath    # also -> evaluation/results/combined_stack.csv

For each provider in turn, everything fitted or counted is taken from the other two providers
(each counting once) and the matches of the held-out one are scored:
  - possession/'s two logistic models (model.fit_logistic on its signals)
  - hmm_v6's speed, flight, gap-distance and width tables (counted as external/score_hmm.py does)
  - the combination's own weight, picked by mean spellF1 on the other two providers
The ball is knocked out to --seen as in external/score_hmm.py: IDSSE and SkillCorner (measured on
labelled frames) down to that share, Metrica as recorded.

The combinations:
  a   hmm_v6's per-frame scores plus w x possession's log P(H), log P(A), log P(D); hmm's decode
  b   possession's log-probabilities plus k x hmm_v6's per-frame scores; possession's decode
  c   two logistic stacks, refitted per fold: team on (possession's log-odds of home, hmm's
      H minus A score), dead on (possession's log-odds of dead, hmm's D minus best-live score);
      possession's decode on the stacked probabilities
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "possession"), os.path.join(ROOT, "external"), ROOT]

METRICS = ("spellF1", "macroF1", "alive/dead", "team")
A_WEIGHTS, B_WEIGHTS = (0.05, 0.1, 0.2), (2.0, 5.0, 10.0)
COMBOS = (["possession", "hmm_v6"] + [f"a w={w:g}" for w in A_WEIGHTS]
          + [f"b k={k:g}" for k in B_WEIGHTS] + ["c stack"])
FAMILIES = {"possession alone": [0], "hmm_v6 alone": [1], "a: possession as a clue in hmm_v6": [2, 3, 4],
            "b: hmm_v6 as a term in possession's decode": [5, 6, 7], "c: logistic stack, possession's decode": [8]}
HMM_CLUES = ("gap", "speed", "flight", "place", "width")


def provider(path):
    return os.path.basename(path).split("_")[0]


def observe(args):
    """One match's per-frame inputs to both models, saved to the scratch folder: possession's
    signals and hmm's clue categories. Nothing here reads a label."""
    path, seen, out = args
    import features
    import model
    import hmm.decode as dec
    import score_hmm as sh
    from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category
    from score import drop_ball
    d = dict(np.load(path))
    halves, pitch, fps = [tuple(h) for h in d["halves"]], tuple(float(v) for v in d["pitch"]), int(d["fps"])
    model.HALVES, model.PITCH, features.PITCH = halves, pitch, pitch
    ball = d["ball"]
    if provider(path) == "idsse":
        ball = drop_ball(ball, seen, fps)
    elif provider(path) == "skillcorner":
        ball = drop_ball(ball, seen, fps, labels=d["labels"])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    np.savez(os.path.join(out, os.path.basename(path)),
             labels=d["labels"], halves=d["halves"],
             dead_x=model.dead_signals(d["players"], d["team"]).astype(np.float32),
             team_x=model.team_signals(dict(players=d["players"], team=d["team"])).astype(np.float32),
             gap=gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away),
             speed=sh.speed_cats(d), flight=sh.flight_cats(d, ball),
             place=ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps), width=sh.width_cats(d))
    return path


def share(cats, sel, n, mirror=None, both=False):
    """One match's p(category) over the frames in `sel`, as external/score_hmm.py counts it."""
    count = lambda s: np.bincount(cats[s & (cats >= 0)], minlength=n).astype(float)  # noqa: E731
    c = count(sel[0])
    if mirror is not None:
        c = c + mirror(count(sel[1 if not both else 0]))
    return c / c.sum()


def hmm_tables(train):
    """hmm_v6's counted tables from these matches' labels, each provider counting once."""
    import hmm.decode as dec
    from hmm.ball import N_FLIGHT_CATS, mirror_flight
    from hmm.clues import N_GAP_DISTANCE, N_SPEED_BINS, N_WIDTH_BINS, mirror_gap_distance
    rev = lambda c: c[::-1]  # noqa: E731
    rows = {}
    for m in train:
        g = m["labels"]
        h, a, dd, live = g == "H", g == "A", g == "D", (g == "H") | (g == "A")
        rows.setdefault(m["prov"], []).append(dict(
            speed_live=share(m["speed"], [live], N_SPEED_BINS), speed_dead=share(m["speed"], [dd], N_SPEED_BINS),
            flight=share(m["flight"], [h, a], N_FLIGHT_CATS, mirror_flight),
            gap_home=share(m["gap"], [h, a], N_GAP_DISTANCE, mirror_gap_distance),
            gap_dead=share(m["gap"], [dd], N_GAP_DISTANCE, mirror_gap_distance, both=True),
            width_home=share(m["width"], [h, a], N_WIDTH_BINS, rev),
            width_dead=share(m["width"], [dd], N_WIDTH_BINS, rev, both=True)))
    mean = lambda k: np.mean([np.mean([r[k] for r in rs], 0) for rs in rows.values()], 0)  # noqa: E731
    floor = lambda p: (p + 1e-3) / (p + 1e-3).sum()  # noqa: E731
    return dict(gap=dec.gap_distance_table(floor(mean("gap_home")), floor(mean("gap_dead"))),
                speed=dec.speed_table(mean("speed_live"), mean("speed_dead")),
                flight=dec.flight_table(floor(mean("flight"))), place=dec.place_table(),
                width=dec.width_table(floor(mean("width_home")), floor(mean("width_dead"))))


def hmm_scores(m, tables):
    """(frames, 3) hmm_v6 per-frame scores for H, A, D: its clues at its own weights."""
    import hmm.decode as dec
    from hmm.model import build_log_emit
    return build_log_emit([(m[k], tables[k], dec.WIDTH_WEIGHT if k == "width" else dec.WEIGHT) for k in HMM_CLUES])


def log_odds(m, fits, tables):
    """possession's log-odds of home and of dead, and hmm's scores, for one match."""
    import model
    lt = np.where(np.isfinite(m["team_x"]).any(1), model.logit(m["team_x"].astype(float), fits["team_fit"]), 0.0)
    return lt, model.logit(m["dead_x"].astype(float), fits["dead_fit"]), hmm_scores(m, tables)


def stack_inputs(lt, ld, E):
    return np.c_[lt, E[:, 0] - E[:, 1]], np.c_[ld, E[:, 2] - E[:, :2].max(1)]


def decode_all(args):
    """Scores of one match under every combination: (combo, metric)."""
    labels, halves, lt, ld, E, stack = args
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import model
    import possession_metrics as pm
    import hmm.decode as dec
    from hmm.model import viterbi as hmm_viterbi
    ls = lambda x: -np.logaddexp(0, -x)  # noqa: E731  log of the logistic
    P = np.c_[ls(-ld) + ls(lt), ls(-ld) + ls(-lt), ls(ld)]  # possession's log P(H), P(A), P(D)
    cost = np.array([[0, model.SWITCH_TEAM, model.SWITCH_DEAD], [model.SWITCH_TEAM, 0, model.SWITCH_DEAD],
                     [model.SWITCH_DEAD, model.SWITCH_DEAD, 0]])
    trans, start = dec.transitions(), np.log(dec.START)
    xt, xd = stack_inputs(lt, ld, E)
    st, sd = model.logit(xt, stack["team"]), model.logit(xd, stack["dead"])
    S = np.c_[ls(-sd) + ls(st), ls(-sd) + ls(-st), ls(sd)]
    emits = ([(P, -cost, np.zeros(3)), (E, trans, start)] + [(E + w * P, trans, start) for w in A_WEIGHTS]
             + [(P + k * E, -cost, np.zeros(3)) for k in B_WEIGHTS] + [(S, -cost, np.zeros(3))])
    truth, out = pm.label_codes(labels), []
    for emit, tr, s0 in emits:
        pred = np.full(len(labels), "X", "<U1")
        for a, b in halves:
            pred[a:b + 1] = np.array(list("HAD"))[hmm_viterbi(emit[a:b + 1], tr, s0)]
        pred[labels == "X"] = "X"
        r = pm.evaluate(truth, pm.label_codes(pred), bootstrap_n=0)
        out.append((r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"]))
    return out


def fit_fold(train):
    """possession's fits and hmm's tables from the training matches, then the stack on top."""
    import model
    provs = [m["prov"] for m in train]
    live = [np.isin(m["labels"], ["H", "A"]) for m in train]
    graded = [m["labels"] != "X" for m in train]
    fits = dict(dead_fit=model.fit_logistic([m["dead_x"][g].astype(float) for m, g in zip(train, graded)],
                                            [m["labels"][g] == "D" for m, g in zip(train, graded)], provs),
                team_fit=model.fit_logistic([m["team_x"][g].astype(float) for m, g in zip(train, live)],
                                            [m["labels"][g] == "H" for m, g in zip(train, live)], provs))
    tables = hmm_tables(train)
    X = [stack_inputs(*log_odds(m, fits, tables)) for m in train]
    stack = dict(team=model.fit_logistic([x[0][g] for x, g in zip(X, live)], [m["labels"][g] == "H" for m, g in zip(train, live)], provs),
                 dead=model.fit_logistic([x[1][g] for x, g in zip(X, graded)], [m["labels"][g] == "D" for m, g in zip(train, graded)], provs))
    return fits, tables, stack


def bar_ath(M):
    """Combination c on BAR-ATH, everything fitted and counted on all the outside matches."""
    import model
    import run as possession_run
    import hmm.decode as dec
    from hmm.model import viterbi as hmm_viterbi
    from starter.baseline import RESULTS_DIR, write_submission
    fits, tables, stack = fit_fold(M)
    F, _ = possession_run.frames()
    x = model.explain(F, fits)
    lo = lambda p: np.log(p.clip(1e-12, 1 - 1e-12)) - np.log((1 - p).clip(1e-12, 1 - 1e-12))  # noqa: E731
    E = hmm_scores(dict(zip(("gap", "place", "speed", "flight", "width"), dec.load_observations())), tables)
    xt, xd = stack_inputs(lo(x["p_home"]), lo(x["p_dead"]), E)
    p_dead = np.where(F["dead"], 1.0, 1 / (1 + np.exp(-model.logit(xd, stack["dead"]))))
    p_home = 1 / (1 + np.exp(-model.logit(xt, stack["team"])))
    labels = np.full(len(p_home), "X", "<U1")
    for a, b in model.HALVES:
        labels[a:b + 1] = np.array(list("HAD"))[model.viterbi(p_home[a:b + 1], p_dead[a:b + 1], model.SWITCH_TEAM, model.SWITCH_DEAD)]
    print("wrote", write_submission(os.path.join(RESULTS_DIR, "combined_stack.csv"), labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seen", type=float, default=0.59)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--bar-ath", action="store_true", help="only write combination c's labels for BAR-ATH")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    scratch = tempfile.mkdtemp(prefix="combined_")
    try:
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(observe, [(p, a.seen, scratch) for p in paths]))
            M = []
            for p in paths:
                z = np.load(os.path.join(scratch, os.path.basename(p)))
                M.append({k: z[k] for k in z.files} | {"prov": provider(p)})
                M[-1]["halves"] = [tuple(h) for h in M[-1]["halves"]]
            if a.bar_ath:
                return bar_ath(M)
            sets = sorted({m["prov"] for m in M})
            held = {}  # provider -> (combo, metric) there, and which combo its fold picks per family
            for test in sets:
                fits, tables, stack = fit_fold([m for m in M if m["prov"] != test])
                print(f"held out {test}: stack team w {stack['team']['w']}, dead w {stack['dead']['w']}", flush=True)
                jobs = [(m["labels"], m["halves"], *log_odds(m, fits, tables), stack) for m in M]
                S = np.array(list(ex.map(decode_all, jobs)))  # match, combo, metric
                by = {s: S[[i for i, m in enumerate(M) if m["prov"] == s]].mean(0) for s in sets}
                train = np.mean([by[s] for s in sets if s != test], 0)
                held[test] = (by[test], {f: c[int(np.argmax(train[c, 0]))] for f, c in FAMILIES.items()})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nball seen {a.seen:.0%}; per held-out provider: " + " / ".join(METRICS))
    for fam in FAMILIES:
        rows = [held[s][0][held[s][1][fam]] for s in sets]
        print(f"{fam}")
        for s, r in zip(sets, rows):
            print(f"  {s:12s} " + "  ".join(f"{x:.3f}" for x in r) + f"   picked: {COMBOS[held[s][1][fam]]}")
        print(f"  {'mean':12s} " + "  ".join(f"{x:.3f}" for x in np.mean(rows, 0)))
    print("\nevery setting, scored on the held-out provider with its fold's fits (spellF1):")
    for c, name in enumerate(COMBOS):
        v = [held[s][0][c, 0] for s in sets]
        print(f"  {name:12s} " + "  ".join(f"{x:.3f}" for x in v) + f"   mean {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
