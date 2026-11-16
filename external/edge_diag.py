"""Diagnostics for stoppage edges: decode v9 held out (as score_fuse), save pred + folded labels."""
import os, sys, glob
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation")); sys.path.insert(0, os.path.join(ROOT, "external"))
import score_fuse as sf
import score_control as sc
import score_hmm as sh
import hmm.decode as dec
from control import CONTROL_SECONDS, control_labels
from control_features import CACHE, match_ball
from hmm.net_clue import log_ratio
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category

def run(name):
    S = sf.STATE
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = S["tables"][k]
    d = sh.load(S["paths"][name]); ball = match_ball(name, d)
    fold = control_labels(d, CONTROL_SECONDS)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t), (sh.flight_cats(d, ball), flight_t)]
    width = sh.width_cats(d)
    loso = S["loso"]
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - np.log(loso["prior_" + name])
    prior = np.mean([S["shares"][o] for o in sh.SETS if o != k], axis=0)
    net = log_ratio(sf.net_probs(name, len(d["halves"]), S["tags"]), d["halves"], len(d["labels"]), prior)
    direct = 0.2 * ratio + 0.3 * net
    pred = dec.decode(clues, dec.transitions(), np.log(dec.START), segments, kickoffs=(), weight=dec.WEIGHT,
                      extra=[(width, width_t, dec.WIDTH_WEIGHT)], direct=direct, min_possession=0)
    pred[d["labels"] == "X"] = "X"
    np.savez_compressed(os.path.join(CACHE, "edge", name + ".npz"), pred=pred, fold=fold, raw=d["labels"],
                        ball=ball, direct=direct, fps=fps, halves=d["halves"])
    return name

if __name__ == "__main__":
    os.makedirs(os.path.join(CACHE, "edge"), exist_ok=True)
    all_paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    state = dict(paths=all_paths, tables=sc.tables(all_paths), shares=sf.label_shares(all_paths),
                 loso=dict(np.load(os.path.join(CACHE, "control_loso.npz"))), tags=["bt", "b2"])
    with Pool(4, initializer=sf._init, initargs=(state,)) as p:
        print(p.map(run, list(all_paths), chunksize=1))
