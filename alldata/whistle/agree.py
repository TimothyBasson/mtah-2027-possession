"""How well audio whistle scores line up with the submission's own stoppages on BAR-ATH.

    python alldata/whistle/agree.py TAGS.npz [--offset0 0 --offset1 0] [--sub FILE]

TAGS.npz from tag.py. Times are mapped to video frames (25 fps) plus a per-half offset (0 for the
tactical video's own audio; 4014 / 26920 for the broadcast's, second half from broadcast frame 85,500).
No ground truth is read: the reference is the submission's decoded dead-ball frames, so this says
whether the detector fires where the model already sees stoppages, more than at random moments.
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WHISTLE = 402


def peaks(score, thr, min_gap=4):
    """Local maxima of score above thr, at least min_gap windows apart."""
    on = np.flatnonzero(score >= thr)
    out = []
    for i in on:
        if out and i - out[-1] < min_gap:
            if score[i] > score[out[-1]]:
                out[-1] = i
            continue
        out.append(i)
    return np.array(out, int)


def to_frames(t, o0, o1, split=85_500):
    f = np.round(t * 25).astype(int)
    return np.where(f < split, f + o0, f + o1)


def runs(lab):
    chg = np.flatnonzero(np.diff(lab) != 0) + 1
    s, e = np.r_[0, chg], np.r_[chg, len(lab)]
    return s, e, lab[s]


def near(fr, pts, lo, hi):
    pts = np.sort(pts)
    i = np.searchsorted(pts, fr + lo)
    return (i < len(pts)) & (pts[np.minimum(i, len(pts) - 1)] <= fr + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags")
    ap.add_argument("--offset0", type=int, default=0)
    ap.add_argument("--offset1", type=int, default=0)
    ap.add_argument("--sub", default="/mnt/project-files/final/alldata_learned_tim_touch.csv")
    a = ap.parse_args()
    z = np.load(a.tags)
    sc = z["p"][:, WHISTLE].astype(float)
    mid = z["start"] + float(z["win"]) / 2
    sub = pd.read_csv(a.sub)["label"].values
    n = len(sub)
    lab = np.select([sub == "D", sub == "X"], [2, 0], 1)
    s, e, l = runs(lab)
    stop = np.array([s[j] for j in range(1, len(s)) if l[j] == 2 and l[j - 1] == 1 and e[j] - s[j] >= 50])
    rest = np.array([s[j] for j in range(1, len(s)) if l[j] == 1 and l[j - 1] == 2])
    graded = lab > 0
    rng = np.random.default_rng(0)
    rand = rng.choice(np.flatnonzero(graded), 20000)
    ch_s, ch_r = near(rand, stop, -50, 75).mean(), near(rand, rest, -75, 25).mean()
    print(f"{len(stop)} decoded stoppages (>= 2 s), {len(rest)} restarts; dead share {np.mean(lab[graded] == 2):.2f}")
    print(f"chance: a random graded frame has a stoppage start within -2..+3 s {ch_s:.2f}, a restart within -3..+1 s {ch_r:.2f}")
    for thr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        pk = peaks(sc, thr)
        fr = to_frames(mid[pk], a.offset0, a.offset1)
        ok = (fr >= 0) & (fr < n)
        fr = fr[ok]
        fr = fr[graded[fr]]
        if not len(fr):
            print(f"thr {thr}: no peaks in graded frames")
            continue
        ns, nr = near(fr, stop, -50, 75).mean(), near(fr, rest, -75, 25).mean()
        rec = near(stop, fr, -75, 50).mean()
        print(f"thr {thr:<5}: {len(fr):4d} peaks; stop start within -2..+3 s {ns:.2f} (x{ns / ch_s:.1f} chance); "
              f"restart within -3..+1 s {nr:.2f} (x{nr / ch_r:.1f}); in dead {np.mean(lab[fr] == 2):.2f}; "
              f"stoppages with a peak {rec:.2f}")


if __name__ == "__main__":
    main()
