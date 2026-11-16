"""Is BAR-ATH's ball early or late against its players? Without labels: at each sharp turn of the
ball's path (a local peak of its change of velocity), when is the nearest player closest to it?
A ball that lags its tracker's players shows the closest approach before the turn. Compared with
outside balls made BAR-ATH-like (smoothed by a symmetric window, so no lag by construction).

    python alldata/learned/lagcheck.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build  # noqa: E402
import feats  # noqa: E402

ROOT = build.ROOT


def offsets(players, bxy, a, b, fps=25, reach=20):
    xy = players[a:b + 1, :, :2].astype(float)
    ball = bxy[a:b + 1].astype(float)
    v = np.zeros_like(ball)
    v[2:-2] = (ball[4:] - ball[:-4]) * fps / 4
    acc = np.full(len(ball), np.nan)
    acc[2:-2] = np.linalg.norm(v[4:] - v[:-4], axis=1)
    d = np.nanmin(np.linalg.norm(xy - ball[:, None], axis=-1), axis=1)
    out, turn = [], []
    for i in range(reach + 6, len(ball) - reach - 6):
        w = acc[i - 6:i + 7]
        if not np.isfinite(w).all() or acc[i] < 2.0 or acc[i] < w.max():
            continue
        seg = d[i - reach:i + reach + 1]
        if not np.isfinite(seg).all() or seg.min() > 2.0:
            continue
        out.append((int(seg.argmin()) - reach) / fps)
        turn.append(acc[i])
    return np.array(out), np.array(turn)


def describe(name, o, t):
    print(f"{name:22s} {len(o):5d} turns: closest approach at median {np.median(o):+.2f} s, mean {o.mean():+.2f} s, "
          f"quartiles {np.percentile(o, 25):+.2f} / {np.percentile(o, 75):+.2f}; velocity change at the turn, median {np.median(t):.1f} m/s, 90th pct {np.percentile(t, 90):.1f}")


def main():
    import warnings
    warnings.simplefilter("ignore")
    z = np.load(os.path.join(ROOT, "data", "cache", "possession_bar_ath_v1.npz"))  # possession/load.py's cache (run.py --features makes it)
    players, ball, halves = z["players"], z["ball"], [tuple(h) for h in z["halves"]]
    o = [offsets(players, ball[:, :2], a, b) for a, b in halves]
    describe("BAR-ATH", np.concatenate([x[0] for x in o]), np.concatenate([x[1] for x in o]))
    for name in ("metrica_1", "metrica_2", "idsse_J03WMX", "idsse_J03WN1", "skillcorner_1874553"):
        d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
        for v in (0, 1):
            bxy = build.ball_version(d, name, v)
            o = [offsets(d["players"], bxy, int(a), int(b)) for a, b in d["halves"]]
            describe(f"{name[:14]} {'BAR-ATH-like' if v else 'as recorded'}"[:22], np.concatenate([x[0] for x in o]), np.concatenate([x[1] for x in o]))


if __name__ == "__main__":
    main()
