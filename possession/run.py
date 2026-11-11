"""Run the model on BAR-ATH and write the submission, or on any match for the viewer.

    uv run python possession/run.py          # -> evaluation/results/possession.csv

frames(match) gives the per-frame inputs for BAR-ATH (match None) or an outside match, with the
ball knocked out to our match's visibility exactly as external/score.py does, so what the viewer
shows is what the score measured.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "external")]
import features  # noqa: E402
import load  # noqa: E402
import model  # noqa: E402
from score import drop_ball, frames as score_frames  # noqa: E402
from starter.baseline import RESULTS_DIR, write_submission  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

SEEN = 0.59  # share of graded frames with a ball in our match, as external/score.py --seen


def frames(match=None, seen=SEEN):
    """(F, labels or None) for a match, with the model's module constants pointed at it."""
    if match is None:
        players, ball, team, halves = load.load()
        pitch, labels = load.PITCH, None
    else:
        z = np.load(os.path.join(ROOT, "data", "external", match + ".npz"))
        players, ball, team, labels = z["players"], drop_ball(z["ball"], seen, model.FPS), z["team"], z["labels"]
        halves, pitch = [tuple(h) for h in z["halves"]], tuple(z["pitch"])
    features.PITCH = model.PITCH = pitch
    model.HALVES = halves
    F = score_frames(players, ball, team, features)
    if match is None:  # graded frames before a kickoff are D (FAQ.md)
        F["dead"] = np.zeros(len(ball), bool)
        for (a, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
            F["dead"][a:kick] = True
    return F, labels


if __name__ == "__main__":
    F, _ = frames()
    path = write_submission(os.path.join(RESULTS_DIR, "possession.csv"), model.predict(F))
    print("wrote", path)
