"""Fit candidate smoothing filters to BAR-ATH's ball spectrum (label-free).

    python alldata/robust/fitfilter.py

Each candidate is applied to raw outside balls (Metrica samples and IDSSE, the sharp ones); the
spectrum of the result is compared with BAR-ATH's between 0.1 and 2 Hz (log power, after
quantising to BAR-ATH's 1e-5 of the pitch). Lower distance is closer.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned"), os.path.join(ROOT, "possession"), HERE]
import feats  # noqa: E402

NSEG = 256
PITCH = np.array([105.3, 68.0])


def runs_seen(xy):
    seen = np.isfinite(xy[:, 0])
    s, t, v = feats._runs(seen)
    return [(i, j) for i, j, k in zip(s, t, v) if k]


def apply(xy, kind, par, fps=25):
    """Filter each seen stretch of the ball on its own. kind: para (half-width, frames), gauss
    (sigma, frames), butter (order, cut-off Hz; forward-backward, so zero lag)."""
    out = np.full_like(xy, np.nan)
    if kind == "para":
        return feats.smooth(xy, par[0] * fps / 25)
    for i, j in runs_seen(xy):
        g = xy[i:j]
        if kind == "gauss":
            out[i:j] = gaussian_filter1d(g, par[0] * fps / 25, axis=0, mode="nearest")
        elif kind == "butter":
            if j - i < 30:
                out[i:j] = g.mean(0)
                continue
            sos = butter(int(par[0]), par[1], fs=fps, output="sos")
            out[i:j] = sosfiltfilt(sos, g, axis=0, padlen=min(3 * fps, j - i - 1))
    return out


def psd(xy, spans, fps=25):
    P, n = 0, 0
    for a, b in spans:
        for i, j in runs_seen(xy[a:b + 1]):
            if j - i < NSEG + 2:
                continue
            g = xy[a + i:a + j]
            if (np.abs(np.diff(g, axis=0)).sum(1) < 1e-4).mean() > 0.2:
                continue
            v = np.diff(g, axis=0) * fps
            for c in (0, 1):
                f, p = welch(v[:, c], fs=fps, nperseg=NSEG, noverlap=NSEG // 2)
                w = max(1, (len(v) - NSEG // 2) // (NSEG // 2))
                P, n = P + p * w, n + w
    return f, P / n


def quant(xy):
    return np.round(xy / PITCH * 1e5) / 1e5 * PITCH


def main():
    import load
    _, ball, _, halves = load.load()
    bar = np.load(os.path.join(HERE, "bar_ball64.npy")) if os.path.exists(os.path.join(HERE, "bar_ball64.npy")) else ball[:, :2].astype(float)
    f, pb = psd(bar, halves)
    band = (f >= 0.1) & (f <= 2.0)
    raws = []
    for name in ("metrica_1", "metrica_2", "metrica_3", "idsse_J03WMX", "idsse_J03WN1", "idsse_J03WOH"):
        d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
        raws.append((name, feats.hide_frozen(d["ball"][:, :2].astype(float)), [(int(h[0]), int(h[1])) for h in d["halves"]]))
    cands = ([("para", (h,)) for h in (16,)] + [("gauss", (s,)) for s in (5, 6, 7, 8, 9, 10)]
             + [("butter", (n, fc)) for n in (2, 3, 4, 5, 6) for fc in (0.6, 0.7, 0.75, 0.8, 0.9, 1.0)])
    sel = [np.argmin(np.abs(f - x)) for x in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 3.0)]
    print(f"{'':22s} dist  | log10 power at " + " ".join(f"{f[i]:5.1f}" for i in sel))
    print(f"{'BAR-ATH':22s}       | " + "               " + " ".join(f"{np.log10(pb[i]):5.2f}" for i in sel))
    res = []
    for kind, par in cands:
        ps = [psd(quant(apply(xy, kind, par)), spans)[1] for _, xy, spans in raws]
        p = np.exp(np.mean(np.log(ps), 0))
        dist = np.sqrt(np.mean((np.log10(p[band]) - np.log10(pb[band])) ** 2))
        res.append((dist, kind, par, p))
        print(f"{kind + ' ' + str(par):22s} {dist:.3f} | " + "               " + " ".join(f"{np.log10(p[i]):5.2f}" for i in sel), flush=True)
    res.sort(key=lambda r: r[0])
    print("closest:", [(r[1], r[2], round(r[0], 3)) for r in res[:5]])


if __name__ == "__main__":
    main()
