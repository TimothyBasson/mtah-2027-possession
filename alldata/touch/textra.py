"""Detected touches as extra input columns for the all-data net (alldata/learned/fit.py --extra).

    python alldata/touch/textra.py --touch t1 [--versions 1 2 3 4]

Writes data/cache/alldata/touch/extra/<touch>/<match>.npz with E<v>_<half>: one row per 5 fps
feature row. Each match takes the detector of the fold in which its provider was held out, so
every match gets what an unseen provider gets (as the players-only net's columns do). Columns,
for home and away: the touch probability now (max over the row's 5 frames), the last touch's and
the next touch's score fading over a second, and whether that team touched last / touches next.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import tuse  # noqa: E402

STEP = 5
THR = 0.3
NAMES = ["tq_H", "tq_A", "tlast_H", "tlast_A", "tnext_H", "tnext_A", "twho_H", "twho_A", "twhon_H", "twhon_A"]


def columns(q, thr=THR, fps=25):
    """q (n, 2) -> (ceil(n / STEP), 10)."""
    n = len(q)
    q = q.astype(np.float32)
    mx = q.copy()
    for d in (1, 2):
        mx[d:] = np.maximum(mx[d:], q[:-d]); mx[:-d] = np.maximum(mx[:-d], q[d:])
    f, tm, sc = tuse.events(q, thr)
    idx = np.arange(n)
    cols = [mx[:, 0], mx[:, 1]]
    last, nxt = [], []
    for k in (0, 1):
        fk, sk = f[tm == k], sc[tm == k]
        if len(fk):
            j = np.searchsorted(fk, idx, "right") - 1
            jl, jn = np.clip(j, 0, len(fk) - 1), np.clip(j + 1, 0, len(fk) - 1)
            l = np.where(j >= 0, sk[jl] * np.exp(-(idx - fk[jl]) / fps), 0)
            x = np.where(j + 1 < len(fk), sk[jn] * np.exp(-(fk[jn] - idx) / fps), 0)
        else:
            l = x = np.zeros(n)
        last.append(l); nxt.append(x)
    cols += last + nxt
    if len(f):
        j = np.searchsorted(f, idx, "right") - 1
        jl, jn = np.clip(j, 0, len(f) - 1), np.clip(j + 1, 0, len(f) - 1)
        okl = (j >= 0) & (idx - f[jl] < 5 * fps)
        okn = (j + 1 < len(f)) & (f[jn] - idx < 5 * fps)
        cols += [okl & (tm[jl] == 0), okl & (tm[jl] == 1), okn & (tm[jn] == 0), okn & (tm[jn] == 1)]
    else:
        cols += [np.zeros(n)] * 4
    return np.stack([np.asarray(c, np.float32) for c in cols], 1)[::STEP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--versions", nargs="+", type=int, default=[1, 2, 3, 4])
    a = ap.parse_args()
    out = os.path.join(C.TOUCH, "extra", a.touch)
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(C.TOUCH, "runs", a.touch, "*", "*.npz"))):
        fold, name = f.split(os.sep)[-2], os.path.basename(f)[:-4]
        if fold not in ("none",) and not name.startswith(fold):
            continue
        z = np.load(f)
        o = {}
        i = 0
        while f"q{a.versions[0]}_{i}" in z.files:
            for v in a.versions:
                o[f"E{v}_{i}"] = columns(z[f"q{v}_{i}"])
            i += 1
        np.savez_compressed(os.path.join(out, name + ".npz"), names=np.array(NAMES), **o)
    print("wrote", out)


if __name__ == "__main__":
    main()
