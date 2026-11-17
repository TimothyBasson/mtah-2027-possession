"""Whistle events from spec.py's features: runs of hops whose band peak stands PROM dB above its
flanks for at least MIN_S seconds (gaps up to GAP_S bridged). Generic settings, the same for any
match: a whistle blast is a tone lasting 0.15 s or more. Times are in seconds of the audio track."""
import numpy as np

PROM, MIN_S, GAP_S = 20.0, 0.15, 0.05


def events(z, prom=PROM, min_s=MIN_S, gap_s=GAP_S):
    hop = float(z["hop"])
    on = z["prom"] >= prom
    # bridge short gaps
    g = int(round(gap_s / hop))
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return np.zeros((0, 4))
    br = np.flatnonzero(np.diff(idx) > g + 1)
    starts, ends = idx[np.r_[0, br + 1]], idx[np.r_[br, len(idx) - 1]] + 1
    keep = (ends - starts) * hop >= min_s
    starts, ends = starts[keep], ends[keep]
    pr = z["prom"]
    cs = np.r_[0, np.cumsum(np.where(on, pr, 0))]
    cn = np.r_[0, np.cumsum(on)]
    mean = (cs[ends] - cs[starts]) / np.maximum(cn[ends] - cn[starts], 1)
    fq = np.array([np.median(z["fpk"][a:b]) for a, b in zip(starts, ends)])
    return np.c_[starts * hop, ends * hop, mean, fq] if len(starts) else np.zeros((0, 4))
