"""Video frames to scan for the ball: every STEP-th frame inside the tracker's ball gaps in the
graded halves, plus every VAL-th frame where the tracker does see the ball (to measure the
detector against the tracker). Writes data/cache/video/targets.npz."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "possession")]
import load

STEP, VAL = 12, 250


def main():
    _, ball, _, halves = load.load_raw()
    gap, val = [], []
    for a, b in halves:
        fr = np.arange(a, b + 1)
        miss = np.isnan(ball[a:b + 1, 0])
        gap.append(fr[miss & (fr % STEP == 0)])
        val.append(fr[~miss & (fr % VAL == 7)])
    gap, val = np.concatenate(gap), np.concatenate(val)
    out = os.path.join(ROOT, "data", "cache", "video", "targets.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, gap=gap, val=val)
    print(len(gap), "gap frames,", len(val), "check frames")


if __name__ == "__main__":
    main()
