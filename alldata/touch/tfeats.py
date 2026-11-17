"""25 fps rows for the touch detector, and its targets: data/cache/alldata/touch/feats/<match>.npz.

    python alldata/touch/tfeats.py [--versions 1 2 3 4 5] [--jobs 4]

Target: the touches of the ball AS RECORDED (hmm.ball.detect_touches, a physical event that means
the same under every provider), with the team of the nearest player. Input: the ball made to
look like BAR-ATH's (alldata/learned/build.ball_version: versions 1-3 parabolic window, 4-6 the
0.7 Hz low-pass of alldata/robust) and the players around it. A touch bends the smoothed path
over about a second and the smoothed ball cuts the corner, so the detector is given the path's
velocity and the nearest three players of each team relative to the ball, and learns the shape.

Columns are built so the team swap is a block swap and mirroring either axis is a sign flip.
"""
import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OUT = os.path.join(C.TOUCH, "feats")
K = 3  # nearest players per team


def learned_build():
    """alldata/learned/build.py by path (a bare `import build` finds external/build.py)."""
    L = os.path.join(C.ROOT, "alldata", "learned")
    sys.path[:0] = [L]
    spec = importlib.util.spec_from_file_location("learned_build", os.path.join(L, "build.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def vel(x, k, fps):
    v = np.zeros_like(x)
    v[k:-k] = (x[2 * k:] - x[:-2 * k]) * (fps / (2 * k))
    return v


def rows(players, team, bxy, fps=25):
    """(n, F) float32 for one half. players (n, P, 3), bxy (n, 2) with NaN where unseen."""
    n = len(bxy)
    seen = np.isfinite(bxy[:, 0])
    idx = np.arange(n)
    if seen.any():
        bi = np.stack([np.interp(idx, idx[seen], bxy[seen, k]) for k in (0, 1)], 1)
    else:
        bi = np.zeros((n, 2))
    v = np.clip(vel(bi, 2, fps), -40, 40)
    cols, names = [], []

    def add(nm, x):
        cols.append(np.asarray(x, np.float32)); names.append(nm)

    add("seen", seen)
    add("vx", v[:, 0] / 10); add("vy", v[:, 1] / 10)
    add("speed", np.log1p(np.linalg.norm(v, axis=1)) / 3)
    for k in (2, 5, 10):  # change of velocity over 0.16, 0.4, 0.8 s
        dv = np.zeros_like(v)
        dv[k:-k] = v[2 * k:] - v[:-2 * k]
        add(f"dvx{k}", np.clip(dv[:, 0], -20, 20) / 5); add(f"dvy{k}", np.clip(dv[:, 1], -20, 20) / 5)
        add(f"dv{k}", np.log1p(np.linalg.norm(dv, axis=1)) / 2)
    pxy = players[:, :, :2].astype(np.float64)
    ok = np.isfinite(pxy[:, :, 0])
    pv = np.stack([vel(np.nan_to_num(pxy[:, :, c]), 2, fps) for c in (0, 1)], -1)
    pv[np.linalg.norm(pv, axis=-1) > 12] = 0
    rel = np.where(ok[:, :, None], pxy - bi[:, None], 1e3)
    dist = np.linalg.norm(rel, axis=-1)
    for ti, tn in ((0, "h"), (1, "a")):
        m = np.flatnonzero(team == ti)
        order = np.argsort(dist[:, m], axis=1)[:, :K]
        for j in range(K):
            if j >= order.shape[1]:
                sel = None
            else:
                sel = m[order[:, j]]
            if sel is None:
                r, d, rv = np.full((n, 2), 30.0), np.full(n, 60.0), np.zeros((n, 2))
            else:
                r, d, rv = rel[idx, sel], dist[idx, sel], pv[idx, sel] - v
                far = d > 60
                r[far], d[far], rv[far] = 30.0, 60.0, 0.0
            add(f"{tn}{j}_x", np.clip(r[:, 0], -15, 15) / 3); add(f"{tn}{j}_y", np.clip(r[:, 1], -15, 15) / 3)
            add(f"{tn}{j}_d", np.log1p(d) / 2)
            add(f"{tn}{j}_near", np.exp(-d / 1.5))
            add(f"{tn}{j}_rvx", np.clip(rv[:, 0], -30, 30) / 10); add(f"{tn}{j}_rvy", np.clip(rv[:, 1], -30, 30) / 10)
    return np.stack(cols, 1), names


def symmetry(names):
    pos = {n: i for i, n in enumerate(names)}
    perm = np.arange(len(names))
    fx, fy = np.ones(len(names), np.float32), np.ones(len(names), np.float32)
    for n, i in pos.items():
        if n[0] in "ha" and n[1].isdigit():
            perm[i] = pos[("a" if n[0] == "h" else "h") + n[1:]]
        if n in ("vx",) or n.startswith("dvx") or n.endswith("_x") or n.endswith("_rvx"):
            fx[i] = -1
        if n in ("vy",) or n.startswith("dvy") or n.endswith("_y") or n.endswith("_rvy"):
            fy[i] = -1
    return perm, fx, fy


def raw_touches(d):
    from hmm.ball import detect_touches
    ball = d["ball"].astype(float)
    dh, da = C.player_dists(d, ball[:, :2])
    t = detect_touches(ball[:, 0], ball[:, 1], dh, da, int(d["fps"]))
    return t["frame"].astype(np.int64), t["team"].astype(np.int8), t["dv"].astype(np.float32)


def target(n, frames, teams, a, b, width=2):
    """(b-a+1,) int8: 0 nothing, 1 home touch, 2 away touch, -1 left out (a touch nobody is beside)."""
    y = np.zeros(b - a + 1, np.int8)
    for f, k in zip(frames, teams):
        if a <= f <= b:
            lo, hi = max(f - a - width, 0), min(f - a + width + 1, b - a + 1)
            y[lo:hi] = k + 1 if k >= 0 else -1
    return y


def one(args):
    path, versions = args
    name = os.path.basename(path)[:-4]
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        return name
    B = learned_build()
    d = dict(np.load(path))
    fps = int(d["fps"])
    assert fps == 25
    fr, tm, dv = raw_touches(d)
    out = dict(tf=fr, tt=tm, tdv=dv, halves=d["halves"])
    names = None
    for v in versions:
        bxy = B.ball_version(d, name, v)
        for i, (a, b) in enumerate((int(h[0]), int(h[1])) for h in d["halves"]):
            F, names = rows(d["players"][a:b + 1], d["team"], bxy[a:b + 1], fps)
            out[f"F{v}_{i}"] = F.astype(np.float16)
            if v == versions[0]:
                out[f"y_{i}"] = target(len(bxy), fr, tm, a, b)
    os.makedirs(OUT, exist_ok=True)
    np.savez(dst + ".tmp.npz", names=np.array(names), versions=np.array(versions), **out)
    os.replace(dst + ".tmp.npz", dst)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    import glob
    paths = sorted(glob.glob(os.path.join(C.ROOT, "data", "external", "*.npz")))
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, [(p, a.versions) for p in paths]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
