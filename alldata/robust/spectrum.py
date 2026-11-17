"""What filter turns a raw outside ball into BAR-ATH's? Label-free: the power spectrum of the
ball's frame-to-frame displacement over stretches where it is seen, BAR-ATH against the outside
providers, and against outside balls passed through candidate filters.

    python alldata/robust/spectrum.py
"""
import os
import sys

import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned"), os.path.join(ROOT, "possession")]
import feats  # noqa: E402

NSEG = 128


def segments(xy, a, b, minlen=NSEG + 2):
    seen = np.isfinite(xy[a:b + 1, 0])
    s, t, v = feats._runs(seen)
    return [xy[a + i:a + j] for i, j, k in zip(s, t, v) if k and j - i >= minlen]


def psd(segs, fps=25):
    """Mean Welch spectrum of the displacement per frame (m/s), x and y pooled."""
    P, n = 0, 0
    for g in segs:
        if (g[1:] == g[:-1]).all(1).mean() > 0.2:  # parked ball
            continue
        v = np.diff(g, axis=0) * fps
        for c in (0, 1):
            f, p = welch(v[:, c], fs=fps, nperseg=NSEG, noverlap=NSEG // 2, detrend="constant")
            w = max(1, (len(v) - NSEG // 2) // (NSEG // 2))
            P, n = P + p * w, n + w
    return f, P / n


def bar():
    import load
    players, ball, team, halves = load.load()
    return players, ball[:, :2].astype(np.float64), halves


def outside(name):
    d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
    return d


def main():
    _, ball, halves = bar()
    f, pb = psd([g for a, b in halves for g in segments(ball, a, b)])
    rows = {"BAR-ATH": pb}
    for name in ("metrica_1", "metrica_2", "metrica_3", "idsse_J03WMX", "idsse_J03WN1", "skillcorner_1874553", "skillcorner_1886347"):
        d = outside(name)
        xy = feats.hide_frozen(d["ball"][:, :2].astype(np.float64))
        hs = [(int(h[0]), int(h[1])) for h in d["halves"]]
        rows[name] = psd([g for a, b in hs for g in segments(xy, a, b)])[1]
    sel = [1, 2, 3, 5, 8, 10, 13, 16, 20, 26, 32, 40, 51, 64]
    print("frequency (Hz)        " + " ".join(f"{f[i]:7.2f}" for i in sel))
    for k, p in rows.items():
        print(f"{k:22s}" + " ".join(f"{np.log10(p[i]):7.2f}" for i in sel))
    print("\nlog10 gain of BAR-ATH over each raw ball (power), normalised at 0.2 Hz")
    for k, p in rows.items():
        if k != "BAR-ATH":
            r = np.log10(pb / p); r -= r[1]
            print(f"{k:22s}" + " ".join(f"{r[i]:7.2f}" for i in sel))


if __name__ == "__main__":
    main()
