"""Score the HMM decoders (hmm/) and the smooth50 baseline on the out-of-sample matches.

    python external/score_hmm.py [--seen 0.59] [--tables]

--seen    as in score.py: knock the IDSSE ball out in random ~2 s runs until it is seen this share
          of frames (ours: 59%). The Metrica samples already lose the ball like ours does and are
          left alone. 1.0 scores IDSSE with the ball always seen.

--tables  also print the gap-distance table counted over both providers, each counting once: the
          constants in hmm/decode.py.

hmm_v3's speed table, hmm_v4's flight table and hmm_v5's gap-distance table are counted from the
labelled external matches, so scoring them on the same matches would be in-sample. Here each
provider is scored with the tables counted on the *other* provider only (IDSSE with Metrica's,
Metrica with IDSSE's). hmm_v2 to v4 are scored at the emission weight they were built with (0.1),
hmm_v5 at the decoder's. Prints aggregates only.

The step between v4 and v5 is scored as well, so that v5's two changes can be told apart, and so
is the variant that is not the default:
    hmm_v4w    v4 at v5's emission weight
    hmm_v5     that, with the gap read together with the distance
    hmm_v5fit  v5 with the ball-place table fitted on each match from its own tracking, without
               labels (hmm/decode.py fit_place; python -m hmm.decode --fit-place)
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import hmm.decode as dec  # noqa: E402
from hmm.ball import N_FLIGHT_CATS, flight_clue, mirror_flight  # noqa: E402
from hmm.clues import (  # noqa: E402
    N_DISTANCE_BINS, N_GAP_BINS, N_GAP_DISTANCE, N_SPEED_BINS, ball_is_usable, ball_place_category,
    gap_category, gap_distance_category, median_player_speed, mirror_gap_distance, speed_category)
from score import drop_ball  # noqa: E402
from starter.baseline import smooth  # noqa: E402

PROVIDERS = ("idsse", "metrica")


def nearest_by_team(players, ball, team):
    """Distance from the ball to the nearest home and nearest away player, NaN where unseen."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        return (np.nanmin(np.where(team[None, :] == 0, d, np.nan), axis=1),
                np.nanmin(np.where(team[None, :] == 1, d, np.nan), axis=1))


def speed_cats(d):
    players = d["players"].astype(float)
    return speed_category(median_player_speed(players[:, :, 0], players[:, :, 1], int(d["fps"])))


def speed_given_state(matches):
    """(live, dead) p(speed bin), the mean over the matches' own labels."""
    rows = []
    for d in matches:
        cats, gt = speed_cats(d), d["labels"]
        rows.append([np.bincount(cats[sel & (cats >= 0)], minlength=N_SPEED_BINS)
                     / np.sum(sel & (cats >= 0))
                     for sel in ((gt != "X") & (gt != "D"), gt == "D")])
    return np.mean(rows, axis=0)


def flight_cats(d, ball):
    """Flight-clue categories for a match, from the ball the model sees."""
    d_home, d_away = nearest_by_team(d["players"], ball, d["team"])
    return flight_clue(ball[:, 0].astype(float), ball[:, 1].astype(float), d_home, d_away,
                       int(d["fps"]))[0]


def flight_given_state(matches, balls):
    """p(flight category | H, in flight), the mean over the matches' own labels. Away frames count
    mirrored into the home row, so the table is symmetric in the teams."""
    rows = []
    for d, ball in zip(matches, balls):
        cats, gt = flight_cats(d, ball), d["labels"]
        count = lambda sel: np.bincount(cats[sel & (cats >= 0)], minlength=N_FLIGHT_CATS)  # noqa
        home = count(gt == "H") + mirror_flight(count(gt == "A"))
        rows.append(home / home.sum())
    home = np.mean(rows, axis=0) + 1e-3
    return home / home.sum()


def gap_distance_given_state(matches, balls):
    """(home, dead) p(gap-distance category | state) on frames with a usable ball, the mean over
    the matches' own labels. Away frames count mirrored into the home row, and dead frames count
    both ways round, so neither row favours a team."""
    rows = []
    for d, ball in zip(matches, balls):
        pitch = tuple(float(v) for v in d["pitch"])
        ball_x, ball_y = ball[:, 0].astype(float), ball[:, 1].astype(float)
        cats = gap_distance_category(ball_is_usable(ball_x, ball_y, pitch),
                                     *nearest_by_team(d["players"], ball, d["team"]))
        gt = d["labels"]
        count = lambda sel: np.bincount(cats[sel & (cats >= 0)],  # noqa: E731
                                        minlength=N_GAP_DISTANCE).astype(float)
        home = count(gt == "H") + mirror_gap_distance(count(gt == "A"))
        dead = count(gt == "D") + mirror_gap_distance(count(gt == "D"))
        rows.append((home / home.sum(), dead / dead.sum()))
    home, dead = np.mean(rows, axis=0) + 1e-3
    return home / home.sum(), dead / dead.sum()


def predict(d, ball, speed_table, flight_table=None, gap_distance_table=None):
    """smooth50, hmm_v2 and hmm_v3 labels for one match, hmm_v4 (v3 plus the flight clue) with a
    flight_table, and hmm_v4w, hmm_v5 and hmm_v5fit with a gap_distance_table as well."""
    players, team = d["players"], d["team"]
    pitch = tuple(float(v) for v in d["pitch"])
    fps = int(d["fps"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    ball_x, ball_y = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = nearest_by_team(players, ball, team)
    gap = gap_category(ball_is_usable(ball_x, ball_y, pitch), d_home, d_away)
    usable = ball_is_usable(ball_x, ball_y, pitch)
    place = ball_place_category(ball_x, ball_y, pitch, dec.CARRY_SECONDS * fps)
    v2_clues = [(gap, dec.gap_table()), (place, dec.place_table())]
    v3_clues = v2_clues + [(speed_cats(d), speed_table)]
    out = {}
    # smooth50 as in starter/baseline.py: nearest team per frame, majority over +-50 frames.
    seen = np.isfinite(ball_x)
    codes = np.where(seen & (d_home <= d_away), 0, np.where(seen & (d_away < d_home), 1, -1))
    s50 = smooth(codes, 50)
    out["smooth50"] = np.full(len(ball), "X", dtype="<U1")
    for lo, hi in segments:
        out["smooth50"][lo:hi] = s50[lo:hi]
    variants = [("hmm_v2", v2_clues), ("hmm_v3", v3_clues)]
    if flight_table is not None:
        flight = flight_cats(d, ball)
        variants.append(("hmm_v4", v3_clues + [(flight, flight_table)]))
    trans, start = dec.transitions(), np.log(dec.START)
    for name, clues in variants:
        out[name] = dec.decode(clues, trans, start, segments, kickoffs=(), weight=dec.WEIGHT_V4)
    if flight_table is not None and gap_distance_table is not None:
        others = [(gap_distance_category(usable, d_home, d_away), gap_distance_table),
                  (speed_cats(d), speed_table), (flight, flight_table)]
        out["hmm_v4w"] = dec.decode(variants[-1][1], trans, start, segments, kickoffs=())
        out["hmm_v5"] = dec.decode(others + [(place, dec.place_table())], trans, start, segments,
                                   kickoffs=())
        live, dead = dec.fit_place(place, others, trans, start, segments, kickoffs=())
        out["hmm_v5fit"] = dec.decode(others + [(place, dec.place_table(live, dead))], trans,
                                      start, segments, kickoffs=())
    return out


def possessions(labels):
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return int(np.sum(np.isin(labels[starts], ["H", "A"])))


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    g, p = gt[gt != "X"], pred[gt != "X"]
    dead_f1 = 2 * np.sum((g == "D") & (p == "D")) / max(np.sum(g == "D") + np.sum(p == "D"), 1)
    return (r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"],
            dead_f1, np.mean(p == "D"), possessions(p))


COLUMNS = "spellF1 macroF1 alive/dead team_acc D_F1 pred_D possessions".split()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seen", type=float, default=0.59)
    ap.add_argument("--tables", action="store_true",
                    help="print the gap-distance table counted over both providers")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    matches = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in paths}
    by_provider = {k: [d for n, d in matches.items() if n.startswith(k)] for k in PROVIDERS}
    tables = {k: dec.speed_table(*speed_given_state(by_provider[o]))
              for k, o in zip(PROVIDERS, PROVIDERS[::-1])}
    balls = {n: drop_ball(d["ball"], a.seen, int(d["fps"])) if n.startswith("idsse") else d["ball"]
             for n, d in matches.items()}
    flight_tables, gap_distance, gap_distance_tables = {}, {}, {}
    for k, o in zip(PROVIDERS, PROVIDERS[::-1]):
        names = [n for n in matches if n.startswith(o)]
        flight_tables[k] = dec.flight_table(flight_given_state(
            [matches[n] for n in names], [balls[n] for n in names]))
        gap_distance[o] = gap_distance_given_state(
            [matches[n] for n in names], [balls[n] for n in names])
        gap_distance_tables[k] = dec.gap_distance_table(*gap_distance[o])
    if a.tables:
        for row, name in enumerate(("H", "D")):
            both = np.mean([gap_distance[k][row] for k in PROVIDERS], axis=0)
            print("p(gap bin, distance bin | {}), each provider counting once; the providers "
                  "differ by at most {:.3f}:".format(name, np.max(np.abs(
                      gap_distance[PROVIDERS[0]][row] - gap_distance[PROVIDERS[1]][row]))))
            for line in both.reshape(N_GAP_BINS, N_DISTANCE_BINS):
                print("    [" + ", ".join("{:.3f}".format(p) for p in line) + "],")

    rows = {}
    for name, d in matches.items():
        provider = name.split("_")[0]
        ball = balls[name]
        gt = d["labels"]
        graded = gt != "X"
        print("\n{}: ball seen {:.0%}, truth D {:.0%}, {} possessions".format(
            name, np.isfinite(ball[graded, 0]).mean(), np.mean(gt[graded] == "D"), possessions(gt)))
        for model, pred in predict(d, ball, tables[provider], flight_tables[provider],
                                   gap_distance_tables[provider]).items():
            pred[~graded] = "X"
            r = score(gt, pred)
            rows.setdefault(model, {})[name] = r
            print("  {:9s} ".format(model) + "  ".join(
                "{} {:.3f}".format(c, v) for c, v in zip(COLUMNS[:5], r[:5]))
                + "  pred D {:.0%}  possessions {}".format(r[5], r[6]))

    for group in PROVIDERS + ("",):
        names = [n for n in matches if n.startswith(group)]
        truth = np.mean([possessions(matches[n]["labels"]) for n in names])
        print("\nmean over {} ({} matches, {:.0f} true possessions):".format(
            group or "all", len(names), truth))
        for model, by_match in rows.items():
            m = np.mean([by_match[n] for n in names], axis=0)
            print("  {:9s} ".format(model) + "  ".join(
                "{} {:.3f}".format(c, v) for c, v in zip(COLUMNS[:5], m[:5]))
                + "  pred D {:.0%}  possessions {:.0f}".format(m[5], m[6]))


if __name__ == "__main__":
    main()
