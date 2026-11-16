"""Tim's HMM (hmm/, v8) as an input to the all-data model: his decoded labels and his control
classifier's posteriors, per frame, for the outside matches and for BAR-ATH. Nothing in hmm/ is
changed; this only calls it.

    python external/control_features.py && python external/train_control.py --loso   # his steps
    python alldata/learned/tim.py            # -> data/cache/alldata/tim/<match>.npz
    python -m hmm.decode                     # his BAR-ATH labels, evaluation/results/hmm_v8.csv
    python alldata/learned/tim.py --bar      # -> data/cache/alldata/tim/BAR-ATH.npz

Outside matches are decoded as external/score_control.py decodes them: the classifier that reads
a match was trained with that match's provider left out. One difference, on purpose: the ball is
the one the match recorded (and that feats.py reads), not score_control.py's ball with random
2 s dropouts, except in the classifier's cached features, which keep his dropouts.
"""

import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external")]
OUT = os.path.join(ROOT, "data", "cache", "alldata", "tim")
STATE = {}


def run(name):
    import hmm.decode as dec
    import score_hmm as sh
    from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = STATE["tables"][k]
    d = sh.load(STATE["paths"][name])
    ball = d["ball"]
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t), (sh.flight_cats(d, ball), flight_t)]
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    post = STATE["loso"][name]
    ratio = np.log(np.clip(post, 1e-6, 1)) - np.log(STATE["loso"]["prior_" + name])
    pred = dec.decode(clues, dec.transitions(), np.log(dec.START), segments, kickoffs=(), extra=extra,
                      direct=dec.CONTROL_WEIGHT * ratio,
                      min_possession=int(round(dec.MIN_POSSESSION_SECONDS * fps)))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, name + ".npz"), labels=pred, post=post.astype(np.float32))
    return name


def bar():
    import pandas as pd
    import hmm.decode as dec
    lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "hmm_v8.csv"))["label"].astype(str).to_numpy(dtype="U1")
    ratio = dec.load_control()  # log posterior - log prior, 0 outside the halves
    import hmm.trees as trees
    prior = np.asarray(trees.load(dec.CONTROL_MODEL)["prior"], float)
    post = np.exp(ratio + np.log(prior))
    post /= post.sum(1, keepdims=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "BAR-ATH.npz"), labels=lab, post=post.astype(np.float32))


def inputs(name, frames_per_half, version=0):
    """[(feature name, (T, 3) probabilities)] for feats.half_features, per half. version: the
    ball his model was run on (build.ball_version); 1.. come from tim_barlike.py."""
    f = os.path.join(OUT, f"{name}.v{version}.npz")
    z = np.load(f if version and os.path.exists(f) else os.path.join(OUT, name + ".npz"))
    out = []
    for fr in frames_per_half:
        lab = z["labels"][fr]
        hard = np.stack([lab == c for c in "HAD"], 1) * 0.9 + 0.05
        out.append([("timlab", hard), ("timctl", z["post"][fr])])
    return out


if __name__ == "__main__":
    if "--bar" in sys.argv:
        bar()
        sys.exit()
    import score_control as sc
    from control_features import CACHE
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    STATE.update(paths=paths, tables=sc.tables(paths), loso=dict(np.load(os.path.join(CACHE, "control_loso.npz"))))
    with Pool(int(os.environ.get("JOBS", 3))) as pool:
        for n in pool.imap(run, list(paths)):
            print(n, flush=True)
