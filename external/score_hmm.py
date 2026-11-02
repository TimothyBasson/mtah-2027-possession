"""Score the HMM decoder (hmm/) and the smooth50 baseline on the out-of-sample matches.

    python external/score_hmm.py [--seen 0.59]

--seen    as in score.py: knock the IDSSE ball out in random ~2 s runs until it is seen this share
          of frames (ours: 59%). The Metrica samples already lose the ball like ours does and are
          left alone. 1.0 scores IDSSE with the ball always seen.

hmm_v3's speed table is counted from the labelled external matches, so scoring it on the same
matches would be in-sample. Here each provider is scored with the table counted on the *other*
provider only (IDSSE with Metrica's, Metrica with IDSSE's). Prints aggregates only.
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
from hmm.clues import (  # noqa: E402
    N_SPEED_BINS, ball_is_usable, ball_place_category, gap_category, median_player_speed,
    speed_category)
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


def predict(d, ball, speed_table):
    """smooth50, hmm_v2 and hmm_v3 labels for one match."""
    players, team = d["players"], d["team"]
    pitch = tuple(float(v) for v in d["pitch"])
    fps = int(d["fps"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    ball_x, ball_y = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = nearest_by_team(players, ball, team)
    gap = gap_category(ball_is_usable(ball_x, ball_y, pitch), d_home, d_away)
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
    for name, clues in (("hmm_v2", v2_clues), ("hmm_v3", v3_clues)):
        out[name] = dec.decode(clues, dec.transitions(), np.log(dec.START), segments, kickoffs=())
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
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    matches = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in paths}
    by_provider = {k: [d for n, d in matches.items() if n.startswith(k)] for k in PROVIDERS}
    tables = {k: dec.speed_table(*speed_given_state(by_provider[o]))
              for k, o in zip(PROVIDERS, PROVIDERS[::-1])}

    rows = {}
    for name, d in matches.items():
        provider = name.split("_")[0]
        ball = drop_ball(d["ball"], a.seen, int(d["fps"])) if provider == "idsse" else d["ball"]
        gt = d["labels"]
        graded = gt != "X"
        print("\n{}: ball seen {:.0%}, truth D {:.0%}, {} possessions".format(
            name, np.isfinite(ball[graded, 0]).mean(), np.mean(gt[graded] == "D"), possessions(gt)))
        for model, pred in predict(d, ball, tables[provider]).items():
            pred[~graded] = "X"
            r = score(gt, pred)
            rows.setdefault(model, {})[name] = r
            print("  {:9s} ".format(model) + "  ".join(
                "{} {:.3f}".format(c, v) for c, v in zip(COLUMNS[:5], r[:5]))
                + "  pred D {:.0%}  possessions {}".format(r[5], r[6]))

    for group in PROVIDERS + ("",):
        names = [n for n in matches if n.startswith(group)]
        print("\nmean over {} ({} matches):".format(group or "all", len(names)))
        for model, by_match in rows.items():
            m = np.mean([by_match[n] for n in names], axis=0)
            print("  {:9s} ".format(model) + "  ".join(
                "{} {:.3f}".format(c, v) for c, v in zip(COLUMNS[:5], m[:5]))
                + "  pred D {:.0%}  possessions {:.0f}".format(m[5], m[6]))


if __name__ == "__main__":
    main()
