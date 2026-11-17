"""The keeper detector as a decoder clue: per frame, log p(state | end features) - log prior, with
the defending / attacking team mapped to H / A, on gated rows only, held for STRIDE frames."""
import numpy as np

from keeper.features import STRIDE


def keeper_direct(p, prior, frames, ends, gate, dteam, n_frames, only=None, floor=1e-3):
    """(n_frames, 3) H, A, D log ratios summed over the two goal ends. only, if given, is a per-row
    mask further limiting where the clue applies (e.g. the ball unseen)."""
    out = np.zeros((n_frames, 3))
    r = np.log(np.clip(p, floor, 1)) - np.log(prior)[None, :]
    use = gate if only is None else gate & only
    for i in np.flatnonzero(use):
        f = frames[i]
        own, other = (0, 1) if dteam[i] == 0 else (1, 0)
        row = np.zeros(3)
        row[own], row[other], row[2] = r[i, 0], r[i, 1], r[i, 2]
        out[f:f + STRIDE] += row
    return out
