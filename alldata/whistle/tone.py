"""A stricter, generic whistle detector: a steady, narrow tone in the usual referee-whistle band.

Per video frame (40 ms hop, 128 ms window, 16 kHz mono): the strongest bin in 3.0-4.2 kHz, its
prominence over the median of 2.8-4.6 kHz, its sharpness (against the loudest bin 80-200 Hz
either side) and whether the pitch held steady (<= 40 Hz since the previous frame). Events: runs
of steady frames with prominence >= 15 dB lasting >= 0.2 s (gaps <= 3 frames), kept when the
peak prominence is >= PEAK dB and the median sharpness >= 10 dB. Settings describe a pea whistle
in general (2.5-4.5 kHz, steady, narrow), not this match; they follow the shape of public
entries' detectors without their band fitted to this recording.

    python alldata/whistle/tone.py AUDIO16k.wav OUT.npz [--offset0 0 --offset1 0]
"""
import argparse

import numpy as np
import pandas as pd

from agree import near, runs
from spec import read

SR, HOP, NFFT = 16000, 640, 2048
BAND, REF = (3000.0, 4200.0), (2800.0, 4600.0)


def frame_scores(x):
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    band, ref = np.flatnonzero((f >= BAND[0]) & (f <= BAND[1])), (f >= REF[0]) & (f <= REF[1])
    df = f[1]
    off = np.r_[-np.arange(int(80 / df), int(200 / df) + 1), np.arange(int(80 / df), int(200 / df) + 1)]
    win = np.hanning(NFFT).astype(np.float32)
    n = (len(x) - NFFT) // HOP
    prom, hz, sharp = (np.zeros(n, np.float32) for _ in range(3))
    for lo in range(0, n, 4096):
        hi = min(n, lo + 4096)
        idx = (np.arange(lo, hi) * HOP)[:, None] + np.arange(NFFT)[None]
        S = 10 * np.log10(np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2 + 1e-12)
        r = np.arange(len(S))
        k = band[np.argmax(S[:, band], axis=1)]
        pk = S[r, k]
        prom[lo:hi] = pk - np.median(S[:, ref], axis=1)
        hz[lo:hi] = f[k]
        sharp[lo:hi] = pk - S[r[:, None], k[:, None] + off[None]].max(axis=1)
    return prom, hz, sharp


def events(prom, hz, sharp, peak):
    hit = (prom >= 15) & np.r_[False, np.abs(np.diff(hz)) <= 40]
    idx = np.flatnonzero(hit)
    if not len(idx):
        return np.zeros(0, int)
    br = np.flatnonzero(np.diff(idx) > 4)
    s, e = idx[np.r_[0, br + 1]], idx[np.r_[br, len(idx) - 1]] + 1
    keep = [a for a, b in zip(s, e) if b - a >= 5 and prom[a:b].max() >= peak and np.median(sharp[a:b]) >= 10]
    return np.array(keep, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--offset0", type=int, default=0)
    ap.add_argument("--offset1", type=int, default=0)
    ap.add_argument("--sub", default="/mnt/project-files/final/alldata_learned_tim_touch.csv")
    a = ap.parse_args()
    prom, hz, sharp = frame_scores(read(a.audio))
    np.savez_compressed(a.out, prom=prom, hz=hz, sharp=sharp)
    sub = pd.read_csv(a.sub)["label"].values
    lab = np.select([sub == "D", sub == "X"], [2, 0], 1)
    s, e, l = runs(lab)
    stop = np.array([s[j] for j in range(1, len(s)) if l[j] == 2 and l[j - 1] == 1 and e[j] - s[j] >= 50])
    rest = np.array([s[j] for j in range(1, len(s)) if l[j] == 1 and l[j - 1] == 2])
    graded = lab > 0
    rand = np.random.default_rng(0).choice(np.flatnonzero(graded), 20000)
    ch_s, ch_r = near(rand, stop, -50, 75).mean(), near(rand, rest, -75, 25).mean()
    print(f"chance: stop start within -2..+3 s {ch_s:.2f}, restart within -3..+1 s {ch_r:.2f}")
    for peak in [20, 24, 28, 32]:
        fr = events(prom, hz, sharp, peak)
        fr = np.where(fr < 85_500, fr + a.offset0, fr + a.offset1)  # broadcast half split (audio frames)
        fr = fr[(fr < len(lab))]
        fr = fr[graded[fr]]
        if not len(fr):
            print(f"peak {peak}: none")
            continue
        ns, nr = near(fr, stop, -50, 75).mean(), near(fr, rest, -75, 25).mean()
        print(f"peak {peak} dB: {len(fr)} events; near stop start {ns:.2f} (x{ns / ch_s:.1f}); near restart {nr:.2f} "
              f"(x{nr / ch_r:.1f}); in dead {np.mean(lab[fr] == 2):.2f}; stoppages with one {near(stop, fr, -75, 50).mean():.2f}")


if __name__ == "__main__":
    main()
