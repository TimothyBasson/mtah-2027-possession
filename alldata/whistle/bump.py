"""Is there any whistle signal at the submission's stoppages? Event-triggered averages.

    python alldata/whistle/bump.py SPEC.npz TAGS.npz [--offset0 0 --offset1 0]

SPEC.npz from spec.py (tone prominence, 10 ms hops), TAGS.npz from tag.py. For every decoded
stoppage start and restart (submission file, no ground truth), averages around it, from -6 s to
+6 s: the share of 10 ms hops whose 1.8-4.6 kHz peak stands >= 20 dB above its flanks, the
strongest such peak per 0.5 s, and the tagger's Whistle probability. A referee's whistle at
stoppages would show as a bump near 0 s against the flat level far from it.
"""
import argparse

import numpy as np
import pandas as pd

from agree import WHISTLE, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("tags")
    ap.add_argument("--offset0", type=int, default=0)
    ap.add_argument("--offset1", type=int, default=0)
    ap.add_argument("--sub", default="/mnt/project-files/final/alldata_learned_tim_touch.csv")
    a = ap.parse_args()
    sp, tg = np.load(a.spec), np.load(a.tags)
    prom = sp["prom"].astype(float)
    hop = float(sp["hop"])
    wh = tg["p"][:, WHISTLE].astype(float)
    sub = pd.read_csv(a.sub)["label"].values
    lab = np.select([sub == "D", sub == "X"], [2, 0], 1)
    s, e, l = runs(lab)
    stop = np.array([s[j] for j in range(1, len(s)) if l[j] == 2 and l[j - 1] == 1 and e[j] - s[j] >= 50])
    rest = np.array([s[j] for j in range(1, len(s)) if l[j] == 1 and l[j - 1] == 2 and s[j] - s[j - 1] >= 50])

    def t_of(frames):  # video frame -> audio seconds
        o = np.where(frames < 100_000, a.offset0, a.offset1)  # split in the halftime gap (video frames 90,652-110,807)
        return (frames - o) / 25.0

    bins = np.arange(-6, 6.01, 0.5)
    for name, ev in [("stoppage starts", stop), ("restarts", rest)]:
        ts = t_of(ev)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            share, top, w = [], [], []
            for t in ts:
                i0, i1 = int((t + lo) / hop), int((t + hi) / hop)
                if i0 < 0 or i1 > len(prom):
                    continue
                seg = prom[i0:i1]
                share.append(np.mean(seg >= 20))
                top.append(seg.max())
                j = int(round((t + (lo + hi) / 2 - 0.5) / 0.5))
                if 0 <= j < len(wh):
                    w.append(wh[j])
            rows.append((lo, np.mean(share), np.mean(top), np.mean(w) * 1000))
        print(f"\n{name} ({len(ev)}): seconds, share of tonal hops, mean max prominence dB, Whistle prob x1000")
        for r in rows:
            print(f"  {r[0]:+5.1f}  {r[1]:.3f}  {r[2]:5.1f}  {r[3]:6.2f}")
    print(f"\nwhole track: share {np.mean(prom >= 20):.3f}, Whistle x1000 {np.mean(wh) * 1000:.2f}")


if __name__ == "__main__":
    main()
