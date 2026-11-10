"""Count flights (hmm/ball.py) on the out-of-sample matches, and what the labels and the current
model (hmm_v3) say during them. Prints aggregates only.

    python external/flights.py [--seen 0.59] [--speed 8 12 16]

The ball is knocked out on IDSSE as in score_hmm.py, for the model and the detector alike. Per
provider, for each flight speed:
    flights/min, mean length, how they end (next touch / ball lost / slowed / capped)
    flight share  of graded frames, and of the frames the labels call H or A
    truth         on in-flight frames whose kicker is known: the kicker's team, the other, D
    hmm_v3        the same shares for the model's labels
    switches      direct H<->A switches, truth's and the model's, and the share inside a flight
    fixable       frames where the truth is the kicker's team and the model says the other team,
                  as a share of all graded frames: what a perfect "kicker holds it in flight"
                  clue could win, at most
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
from hmm.ball import FLIGHT_SPEED, detect_touches, flights  # noqa: E402
from score import drop_ball  # noqa: E402
from score_hmm import PROVIDERS, nearest_by_team, predict, speed_given_state  # noqa: E402
import hmm.decode as dec  # noqa: E402

ENDS = ("touch", "lost", "slow", "cap")


def flight_stats(labels, pred, ball, players, team, fps, graded, min_speed):
    d_home, d_away = nearest_by_team(players, ball, team)
    t = detect_touches(ball[:, 0], ball[:, 1], d_home, d_away, fps)
    f = flights(t, fps, min_speed)
    keep = graded[f["start"]]
    fly = f["in_flight"] & graded
    known = fly & (f["kicker"] >= 0)
    kick = np.where(f["kicker"] == 0, "H", "A")
    other = np.where(f["kicker"] == 0, "A", "H")
    live = np.isin(labels, ["H", "A"])
    r = {"per_min": keep.sum() / (graded.sum() / fps / 60),
         "length_s": np.mean((f["stop"] - f["start"])[keep]) / fps,
         "share": fly.sum() / graded.sum(),
         "share_live": np.sum(fly & live) / np.sum(live),
         "known": known.sum() / max(fly.sum(), 1)}
    r.update({"end_" + e: np.mean(f["end"][keep] == k) for k, e in enumerate(ENDS)})
    for name, lab in (("truth", labels), ("model", pred)):
        r[name + "_kicker"] = np.mean(lab[known] == kick[known])
        r[name + "_other"] = np.mean(lab[known] == other[known])
        r[name + "_dead"] = np.mean(lab[known] == "D")
    for name, lab in (("truth", labels), ("model", pred)):
        flip = np.zeros(len(lab), dtype=bool)
        alive = np.isin(lab, ["H", "A"])
        flip[1:] = alive[1:] & alive[:-1] & (lab[1:] != lab[:-1])
        r[name + "_flips"] = np.sum(flip & graded)
        r[name + "_flips_in_flight"] = np.sum(flip & fly)
    r["fixable"] = np.sum(known & (labels == kick) & (pred == other)) / graded.sum()
    r["broken"] = np.sum(known & (labels == other) & (pred == other)) / graded.sum()
    return r


def show(name, rows):
    m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print(f"  {name}: {m['per_min']:.1f} flights/min, {m['length_s']:.2f} s long, ends " +
          " ".join(f"{e} {m['end_' + e]:.0%}" for e in ENDS) +
          f"; in flight {m['share']:.1%} of graded, {m['share_live']:.1%} of live, kicker known "
          f"{m['known']:.0%}")
    for who in ("truth", "model"):
        print(f"     {who}: kicker {m[who + '_kicker']:.0%}  other {m[who + '_other']:.0%}  "
              f"D {m[who + '_dead']:.0%}")
    print("     H<->A switches inside flights: truth {:.0%} of its {:.0f}, "
          "model {:.0%} of its {:.0f}"
          .format(m["truth_flips_in_flight"] / m["truth_flips"], m["truth_flips"],
                  m["model_flips_in_flight"] / m["model_flips"], m["model_flips"]))
    print(f"     fixable {m['fixable']:.2%} of graded frames (model other, truth kicker); "
          f"at risk {m['broken']:.2%} (both other)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seen", type=float, default=0.59)
    ap.add_argument("--speed", type=float, nargs="+", default=[FLIGHT_SPEED])
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    matches = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in paths}
    by_provider = {k: [d for n, d in matches.items() if n.startswith(k)] for k in PROVIDERS}
    tables = {k: dec.speed_table(*speed_given_state(by_provider[o]))
              for k, o in zip(PROVIDERS, PROVIDERS[::-1])}
    prepared = []
    for name, d in matches.items():
        provider = name.split("_")[0]
        ball = drop_ball(d["ball"], a.seen, int(d["fps"])) if provider == "idsse" else d["ball"]
        pred = predict(d, ball, tables[provider])["hmm_v3"]
        prepared.append((provider, d, ball.astype(float), pred))
    for speed in a.speed:
        print(f"flights off a touch at >= {speed} m/s")
        for provider in PROVIDERS:
            rows = [flight_stats(d["labels"], pred, ball, d["players"].astype(float), d["team"],
                                 int(d["fps"]), d["labels"] != "X", speed)
                    for p, d, ball, pred in prepared if p == provider]
            show(provider, rows)


if __name__ == "__main__":
    main()
