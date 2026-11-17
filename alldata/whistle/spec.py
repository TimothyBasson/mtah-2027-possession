"""Per-10 ms whistle features from a 16 kHz mono WAV (numpy only).

A referee whistle is a strong, narrow tone (about 2-4.5 kHz, often with a warble) standing well
above the crowd and commentary around it. For each 10 ms hop: the peak bin in 1.8-4.6 kHz, its
level above the median of the bins 150-600 Hz either side (prominence, dB), and the peak's
frequency. Nothing here is tied to a match.
"""
import sys
import wave

import numpy as np

SR, NFFT, HOP = 16000, 1024, 160
LO, HI = 1800, 4600


def read(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def features(x, chunk=200_000):
    win = np.hanning(NFFT).astype(np.float32)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    band = np.flatnonzero((freqs >= LO) & (freqs <= HI))
    df = freqs[1]
    near, far = int(150 / df), int(600 / df)
    nh = (len(x) - NFFT) // HOP + 1
    prom = np.zeros(nh, np.float32)
    fpk = np.zeros(nh, np.float32)
    lvl = np.zeros(nh, np.float32)
    for s in range(0, nh, chunk):
        e = min(nh, s + chunk)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(NFFT)[None]
        P = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2 + 1e-12
        L = 10 * np.log10(P)
        k = band[np.argmax(L[:, band], axis=1)]
        rows = np.arange(e - s)
        # median of the flanks: bins near..far below and above the peak
        offs = np.r_[-np.arange(near, far), np.arange(near, far)]
        fl = np.median(L[rows[:, None], np.clip(k[:, None] + offs[None], 0, L.shape[1] - 1)], axis=1)
        pk = np.max(L[rows[:, None], np.clip(k[:, None] + np.arange(-1, 2)[None], 0, L.shape[1] - 1)], axis=1)
        prom[s:e] = pk - fl
        fpk[s:e] = freqs[k]
        lvl[s:e] = pk
    return prom, fpk, lvl


if __name__ == "__main__":
    x = read(sys.argv[1])
    prom, fpk, lvl = features(x)
    np.savez_compressed(sys.argv[2], prom=prom, fpk=fpk, lvl=lvl, hop=HOP / SR)
    print(sys.argv[2], len(prom), np.percentile(prom, [50, 90, 99, 99.9]))
