"""Per-player paths for the ball-holder network: data/cache/alldata/players/<match>.npz.

    python alldata/holder/pcache.py [--jobs 4]

Per half: P (T, K, 4) float16, each player's x, y (metres from the centre spot, home defending
x < 0: the frame of alldata/learned/feats.py) and velocity, at the rows of the feature cache
(5 fps), NaN where the player is not tracked; and the team of each of the K columns. The ball is
not stored here: the network reads it from the feature rows, so one cache serves every simulated
version of the ball.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path += [os.path.join(ROOT, "deep")]
import data as D  # noqa: E402

OUT = os.path.join(ROOT, "data", "cache", "alldata", "players")


def half_paths(players, team, a, b, pitch, fps=25, k=2):
    """(T, K', 4) and the teams of the K' columns tracked at some point of the half."""
    xy = players[a:b + 1, :, :2].astype(np.float32) - np.array([pitch[0] / 2, pitch[1] / 2], np.float32)
    if not D.home_defends_left(players, team, a, b):
        xy = -xy
    keep = np.flatnonzero(np.isfinite(xy[:, :, 0]).any(0))
    xy = xy[:, keep]
    v = np.full_like(xy, np.nan)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2 * k))
    v[np.linalg.norm(v, axis=-1) > D.MAX_SPEED] = np.nan
    v = np.where(np.isfinite(xy) & ~np.isfinite(v), 0, v)
    sel = np.arange(0, len(xy), D.STEP)
    return np.concatenate([xy[sel], v[sel]], -1).astype(np.float16), team[keep].astype(np.int8)


def one(path):
    name = os.path.basename(path)[:-4]
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        return name
    d = np.load(path)
    out = {}
    for i, (a, b) in enumerate(d["halves"]):
        out[f"P_{i}"], out[f"team_{i}"] = half_paths(d["players"], d["team"], int(a), int(b), d["pitch"], int(d["fps"]))
    os.makedirs(OUT, exist_ok=True)
    np.savez(dst + ".tmp.npz", **out)
    os.replace(dst + ".tmp.npz", dst)
    return name


def bar_ath():
    """BAR-ATH's paths, beside its feature rows (alldata/learned/run.py --features)."""
    sys.path.append(os.path.join(ROOT, "possession"))
    import load
    players, _, team, halves = load.load()
    out = {}
    for i, (a, b) in enumerate(halves):
        out[f"P_{i}"], out[f"team_{i}"] = half_paths(players, team, a, b, load.PITCH, load.FPS)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "BAR-ATH.npz"), **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--bar", action="store_true")
    a = ap.parse_args()
    if a.bar:
        bar_ath()
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(one, D.external_paths()):
                print(r, flush=True)
