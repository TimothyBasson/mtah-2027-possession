"""Label BAR-ATH from saved probabilities and write the submission.

    python alldata/physics/train.py --run <run> --fold none     # trains on all 30, predicts BAR-ATH
    python alldata/physics/run.py --run <run> [--tags] [--out alldata_physics.csv]

Decodes each half, sets graded frames before each kick-off to D, optionally draws a stoppage at
Smart Tagging set pieces the model plays through (stops/tags.py), writes
evaluation/results/<out>, validates it, and prints the public-window score as a check only.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import decode as DC  # noqa: E402
from train import pred_dir  # noqa: E402

DECODE = dict(cost=4.0, min_team=1.0, min_dead=1.0)


def label(probs, halves, n, tags=False, **dec):
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    out = np.full(n, "X", "<U1")
    if tags:
        sys.path.insert(0, os.path.join(C.ROOT, "stops"))
        import tags as TG
        starts = TG.set_pieces()
    for p, (a, b) in zip(probs, halves):
        m = b - a + 1
        path = DC.decode_half(p, m, **dec)
        if tags:
            lp = np.log(np.clip(DC.upsample(p, m), 1e-6, 1))
            lp, hit = TG.apply(lp, path, a, starts)
            if hit:
                path = DC.absorb(DC.viterbi(lp, DC.cost_matrix(dec["cost"], dec.get("cost_dead"))),
                                 {0: dec["min_team"] * 25, 1: dec["min_team"] * 25, 2: dec["min_dead"] * 25})
                print("stoppage drawn for the set pieces tagged at frames", hit)
        out[a:b + 1] = DC.LETTERS[path]
    for (a, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        out[a:kick] = "D"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="one run, or several whose probabilities are averaged")
    ap.add_argument("--out", default="alldata_physics.csv")
    ap.add_argument("--tags", action="store_true")
    a = ap.parse_args()
    from starter.baseline import RESULTS_DIR, write_submission
    from games import half_video_blocks, source_frames, GAME
    halves = [(s, e) for _, s, e in half_video_blocks(GAME)]
    zs = [np.load(os.path.join(pred_dir(r), "bar_ath.npz")) for r in a.run]
    probs = [np.mean([z[f"p{h}"] for z in zs], 0) for h in range(len(halves))]
    pred = label(probs, halves, source_frames(), a.tags, **DECODE)
    path = write_submission(os.path.join(RESULTS_DIR, a.out), pred)
    print("wrote", path)
    subprocess.run([sys.executable, os.path.join(C.ROOT, "evaluation", "validate_submission.py"), path], check=False)


if __name__ == "__main__":
    main()
