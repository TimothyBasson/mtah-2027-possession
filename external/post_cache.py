"""Cache what the decoder sees on each external match, for experiments on what happens after the
classifier (external/post_sweep.py): the v6 clues' log emissions, the held-out control ratio, the
labels (raw and control-folded), the halves and the touches on the ball the model is shown.

    python external/post_cache.py [--loso control_loso.npz]     # -> data/post/<match>.npz
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "external"))
import hmm.decode as dec  # noqa: E402
import score_control as sc  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from control_features import CACHE, match_ball  # noqa: E402
from hmm.ball import detect_touches  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402
from hmm.model import build_log_emit  # noqa: E402

OUT = os.path.join(ROOT, "data", "post")
STATE = {}


def _init(state):
    STATE.update(state)


def run(name):
    paths, tabs, loso = STATE["paths"], STATE["tables"], STATE["loso"]
    speed_t, flight_t, gap_t, width_t = tabs[name.split("_")[0]]
    d = sh.load(paths[name])
    ball = match_ball(name, d)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t),
             (sh.flight_cats(d, ball), flight_t)]
    emit = build_log_emit([(o, t, dec.WEIGHT) for o, t in clues]
                          + [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)])
    out = dict(emit=emit.astype(np.float32), labels=d["labels"], folded=control_labels(d),
               fps=fps, halves=np.array([(int(a), int(b) + 1) for a, b in d["halves"]]),
               d_home=d_home.astype(np.float32), d_away=d_away.astype(np.float32))
    for key, z in loso.items():
        out["ratio_" + key] = (np.log(np.clip(z[name], 1e-6, 1))
                               - np.log(z["prior_" + name])).astype(np.float32)
    touches = detect_touches(bx, by, d_home, d_away, fps)
    out["touch_frame"], out["touch_team"] = touches["frame"], touches["team"]
    np.savez(os.path.join(OUT, name + ".npz"), **out)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loso", default="control_loso.npz", help="comma-separated posterior files")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    paths = {os.path.basename(p)[:-4]: p
             for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    loso = {f[:-4]: dict(np.load(os.path.join(CACHE, f))) for f in a.loso.split(",")}
    os.makedirs(OUT, exist_ok=True)
    state = dict(paths=paths, tables=sc.tables(paths), loso=loso)
    with Pool(a.jobs, initializer=_init, initargs=(state,)) as pool:
        for n in pool.imap_unordered(run, list(paths)):
            print(n, flush=True)


if __name__ == "__main__":
    main()
