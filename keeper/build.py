"""Keeper-end feature rows for the 30 external matches, two balls each:
  score: the scorer's ball (external/control_features.match_ball), what the decoder sees held out
  hands: that ball also hidden in hands inside the boxes (hide_in_hands), as BAR-ATH loses it
-> data/cache/keeper/<match>.npz"""
import glob, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
from control import CONTROL_SECONDS, control_labels  # noqa
from control_features import match_ball  # noqa
from keeper.features import end_features, hide_in_hands  # noqa

OUT = os.path.join(ROOT, "data", "cache", "keeper")


def build(path):
    name = os.path.basename(path)[:-4]
    out = os.path.join(OUT, name + ".npz")
    if os.path.exists(out):
        return name
    d = np.load(path)
    lab = control_labels(d, CONTROL_SECONDS)
    ball = match_ball(name, d).astype(float)
    pl = d["players"]
    px, py = pl[:, :, 0], pl[:, :, 1]
    pitch = tuple(float(v) for v in d["pitch"])
    fps = int(d["fps"])
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    res = {}
    hx, hy, hide = hide_in_hands(ball[:, 0], ball[:, 1], px, py, pitch)
    for tag, (bx, by) in (("score", (ball[:, 0], ball[:, 1])), ("hands", (hx, hy))):
        X, names, fr, ends, gate, y, dteam = end_features(px, py, d["team"], bx, by, pitch, fps,
                                                          halves, lab)
        res.update({tag + "_X": X, tag + "_gate": gate})
    np.savez_compressed(out, names=np.array(names), frames=fr, ends=ends, y=y, dteam=dteam,
                        hidden_share=hide.mean(), **res)
    return name


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    if len(sys.argv) > 1:
        paths = [p for p in paths if os.path.basename(p).startswith(sys.argv[1])]
    with Pool(int(os.environ.get("JOBS", 2))) as pool:
        for n in pool.imap_unordered(build, paths):
            print(n, flush=True)
