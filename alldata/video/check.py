"""The detector against the tracker on the check frames (tracker sees the ball): the only check
BAR-ATH allows without labels. Reports, per confidence floor, the share of check frames with a
pick and how far picks are from the tracker's ball."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "possession")]
import load, ballfill


def main():
    _, ball, _, _ = load.load_raw()
    t = np.load(os.path.join(ROOT, "data/cache/video/targets.npz"))
    val = np.intersect1d(t["val"], ballfill.scanned_frames())
    print("check frames scanned", len(val))
    for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        b = ballfill.picks(c)
        f = np.array([k for k in b if k in set(val)], int)
        if not len(f):
            continue
        d = np.linalg.norm(np.array([b[k][:2] for k in f]) - ball[f, :2], axis=1)
        print(f"conf>={c}: picked {len(f)/len(val):.2f} of check frames; of picks within 2 m {np.mean(d<2):.2f}, "
              f"within 5 m {np.mean(d<5):.2f}, median {np.median(d):.1f} m")


if __name__ == "__main__":
    main()
