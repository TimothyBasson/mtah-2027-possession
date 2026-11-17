"""BAR-ATH: v9 (as hmm.decode, hold on) plus the keeper clue at weight W from hmm/models/keeper.pkl.
    python keeper/barath_decode.py W[,W2...] [--gate all|blind]
Writes evaluation/results/hmm_v9_keeper<W>[_blind].csv (moved out of results/ by the caller)."""
import argparse, os, pickle, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hmm.decode as dec  # noqa
from keeper.barath import load  # noqa
from keeper.clue import keeper_direct  # noqa
from keeper.features import end_features  # noqa


def keeper_clue(model_path=os.path.join(ROOT, "hmm", "models", "keeper.pkl"), gate="all"):
    d = load()
    m = pickle.load(open(model_path, "rb"))
    X, names, fr, ends, g, y, dteam = end_features(d["px"], d["py"], d["team"], d["bx"], d["by"],
                                                   tuple(d["pitch"]), int(d["fps"]),
                                                   [tuple(h) for h in d["halves"]])
    cols = [names.index(n) for n in m["names"]]
    p = m["model"].predict_proba(X[:, cols])
    only = None if gate == "all" else X[:, names.index("seen")] < 0.5
    return keeper_direct(p, m["prior"], fr, ends, g, dteam, len(d["px"]), only=only), (p, fr, ends, g, dteam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--gate", default="all")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "logs_kd", "barath"))
    a = ap.parse_args()
    gap, place, speed, flight, width = dec.load_observations()
    segments = dec.graded_segments()
    clues = [(gap, dec.gap_distance_table()), (place, dec.place_table(dec.PLACE_GIVEN_LIVE, dec.PLACE_GIVEN_DEAD)),
             (speed, dec.speed_table()), (flight, dec.flight_table())]
    extra = [(width, dec.width_table(), dec.WIDTH_WEIGHT)]
    base = dec.CONTROL_WEIGHT * dec.load_control() + dec.NET_WEIGHT * dec.load_net(tuple(dec.NET_MODELS))
    kd, _ = keeper_clue(gate=a.gate)
    d = load()
    os.makedirs(a.outdir, exist_ok=True)
    for w in [float(x) for x in a.weights.split(",")]:
        labels = dec.decode(clues, dec.transitions(), np.log(dec.START), segments, weight=dec.WEIGHT,
                            extra=extra, direct=base + w * kd,
                            min_possession=int(round(dec.MIN_POSSESSION_SECONDS * dec.FPS)))
        labels = dec.hold_restarts(labels, d["bx"], d["by"], dec.FPS)
        path = os.path.join(a.outdir, "hmm_v9_keeper{:g}{}.csv".format(w, "" if a.gate == "all" else "_blind"))
        dec.write_submission(path, labels)
        print(path, "D share", np.mean(labels[labels != "X"] == "D"))


if __name__ == "__main__":
    main()
