"""Build hmm/control.py joint features for every external match, both team orientations, cached.

    python external/control_features.py      # -> data/cache/control_v8_<match>.npz

Each file holds the features on every STRIDE-th frame (X, Xs with the teams swapped), the frames
they sit on and the names of the columns.

The ball is the one external/score_hmm.py scores with (--seen 0.59): IDSSE and SkillCorner topped
up with ~2 s dropouts, Metrica as recorded. So the classifier trains on the same kind of missing
ball the decoder is scored on, and that BAR-ATH has.
"""

import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external"))
from hmm.control import joint_features  # noqa: E402
from score import drop_ball  # noqa: E402

SEEN = 0.59
CACHE = os.environ.get("CONTROL_CACHE", os.path.join(ROOT, "data", "cache"))


def match_ball(name, d, seen=SEEN):
    if name.startswith("idsse"):
        return drop_ball(d["ball"], seen, int(d["fps"]))
    if name.startswith("skillcorner"):
        return drop_ball(d["ball"], seen, int(d["fps"]), labels=d["labels"])
    return d["ball"]


def cache_path(name):
    return os.path.join(CACHE, "control_v8_{}.npz".format(name))


def build(path):
    name = os.path.basename(path)[:-4]
    out = cache_path(name)
    if os.path.exists(out):
        return name
    d = np.load(path)
    ball = match_ball(name, d)
    players = d["players"].astype(np.float32)
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    args = (players[:, :, 0], players[:, :, 1], d["team"], ball[:, 0].astype(float),
            ball[:, 1].astype(float), tuple(float(v) for v in d["pitch"]), int(d["fps"]), halves)
    X, names, frames = joint_features(*args)
    Xs = joint_features(*args, swap=True)[0]
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, X=X, Xs=Xs, names=np.array(names), frames=frames,
                        labels=d["labels"])
    return name


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    if len(sys.argv) > 1:
        paths = [p for p in paths if any(k in p for k in sys.argv[1:])]
    for p in paths:
        t = time.time()
        print(build(p), "%.0f s" % (time.time() - t), flush=True)
