"""Apply the segment corrector to a BAR-ATH file made by alldata/learned/run.py.

    python alldata/learned/run.py                       # the base file and its probabilities
    python alldata/corrector/run.py [--tags tpsN tpsNs1] [--model segmix] [--thr 0.7] [--out alldata_learned_tim_corr.csv]

Reads the base nets' cached probabilities, decodes them as the base file does, finds candidate
segments (segs.py), and relabels those the corrector scores above the threshold. Reads BAR-ATH's
players and ball. No ground truth is read.
"""

import argparse
import glob
import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE]
sys.path += [os.path.join(ROOT, "possession"), ROOT]
import segfit as SF
import segs

LETTERS = np.array(list("HAD"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["tpsN", "tpsNs1"])
    ap.add_argument("--model", default="segmix")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--out", default="alldata_learned_tim_corr.csv")
    a = ap.parse_args()
    import load

    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    players, ball, team, halves = load.load()
    z = np.load(os.path.join(ROOT, "data", "cache", "alldata", "bar_ath", "+".join(a.tags) + ".npz"))
    models = [lgb.Booster(model_file=f) for f in sorted(glob.glob(os.path.join(HERE, "models", a.model + "_s*.txt")))]
    pred = np.full(len(players), "X", "<U1")
    nflip = 0
    for i, (s0, s1) in enumerate(halves):
        path, spans, to, rows = segs.half_candidates(players, ball[:, :2], team, s0, s1, load.PITCH, load.FPS, z[f"p{i}"], "a")
        if len(spans):
            p = np.mean([m.predict(rows) for m in models], 0)
            keep = SF.choose(spans, p, a.thr)
            for (s, e), y, q, kp in zip(spans, to, p, keep):
                if kp:
                    path[s:e] = y
                    nflip += 1
                    print(f"  half {i + 1}: frames {s0 + s}-{s0 + e} ({(e - s) / load.FPS:.1f} s) -> {'HA'[y]}  p {q:.2f}")
        pred[s0:s1 + 1] = LETTERS[path]
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print(nflip, "segments relabelled")
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
