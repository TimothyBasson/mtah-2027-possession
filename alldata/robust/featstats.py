"""Do the model's own inputs look the same on BAR-ATH and on a simulated outside ball?
Label-free: distribution of the ball columns of feats.py over frames where the ball is seen.

    python alldata/robust/featstats.py
"""
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "alldata", "learned"), os.path.join(ROOT, "possession")]
import feats  # noqa: E402


def stats(F, names, fps5=5):
    c = {n: i for i, n in enumerate(names)}
    seen = F[:, c["seen"]] > 0.5
    acc = np.expm1(F[seen, c["acc"]] * 3)
    sp = np.expm1(F[seen, c["speed"]] * 3)
    ago = F[seen, c["t_ago"]] * 10
    relv = np.minimum(F[seen, c["h_relv"]], F[seen, c["a_relv"]]) * 10
    mv = sp > 2
    return [seen.mean(), np.median(sp[mv]), np.percentile(sp[mv], 90), np.median(acc[mv]), np.percentile(acc[mv], 90), np.percentile(acc[mv], 99),
            (acc > feats.TOUCH_DV).mean(), np.median(ago), (ago < 1).mean(), np.median(relv), np.percentile(relv, 90)]


def main():
    import load
    players, ball, team, halves = load.load()
    hdr = "seen | speed med p90 | acc med p90 p99 | acc>thr | touch ago med, <1s | relv med p90"
    print(f"{'':30s}{hdr}")
    Fs = [feats.half_features(players, ball[:, :2], team, a, b, load.PITCH, load.FPS)[0] for a, b in halves]
    for i, F in enumerate(Fs):
        print(f"{'BAR-ATH half ' + str(i + 1):30s}" + " ".join(f"{x:6.2f}" for x in stats(F, feats.NAMES)))
    for name in ("metrica_1", "metrica_2", "metrica_3", "idsse_J03WMX", "idsse_J03WN1", "skillcorner_1874553"):
        d = dict(np.load(os.path.join(ROOT, "data", "external", name + ".npz")))
        for style in ("para", "lp"):
            rng = np.random.default_rng(zlib.crc32(f"{name} stats".encode()))
            out = []
            for a, b in ((int(h[0]), int(h[1])) for h in d["halves"]):
                bxy = d["ball"][:, :2].astype(np.float32).copy()
                bxy[a:b + 1] = feats.bar_like(bxy[a:b + 1], d["labels"][a:b + 1], rng, int(d["fps"]), style=style)
                out.append(feats.half_features(d["players"], bxy, d["team"], a, b, d["pitch"], int(d["fps"]))[0])
            print(f"{name[:18] + ' ' + style:30s}" + " ".join(f"{x:6.2f}" for x in stats(np.concatenate(out), feats.NAMES)))


if __name__ == "__main__":
    main()
