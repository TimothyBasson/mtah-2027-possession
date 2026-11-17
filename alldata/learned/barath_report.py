"""Label-free look at BAR-ATH files, plus the public 5 minutes possession by possession (a check).

    python alldata/learned/barath_report.py name=path.csv [name=path.csv ...]

Per half: possessions, their lengths, share dead; agreement with the first file split by whether
the ball is seen. Then each true possession of the public window against each file.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "possession"), os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external"), ROOT]
import load  # noqa: E402
import possession_metrics as pm  # noqa: E402


def runs(x):
    e = np.flatnonzero(x[1:] != x[:-1]) + 1
    return np.r_[0, e], np.r_[e, len(x)]


def main():
    files = dict(a.split("=") for a in sys.argv[1:])
    P = {k: pd.read_csv(f).label.astype(str).values for k, f in files.items()}
    players, ball, team, halves = load.load()
    seen = np.isfinite(ball[:, 0])
    first = next(iter(P))
    for k, p in P.items():
        for hi, (a, b) in enumerate(halves):
            x = p[a:b + 1]
            s, t = runs(x)
            L = (t - s) / 25
            poss = np.isin(x[s], ["H", "A"])
            dead = x[s] == "D"
            live_min = np.isin(x, ["H", "A"]).sum() / 25 / 60
            print(f"{k:10s} half {hi + 1}: {poss.sum():3d} possessions, median {np.median(L[poss]):4.1f} s, "
                  f"under 2 s {np.mean(L[poss] < 2):4.0%}, under 5 s {np.mean(L[poss] < 5):4.0%}, "
                  f"{poss.sum() / live_min:4.1f} per live minute, dead {np.mean(x == 'D'):4.0%} in {dead.sum()} stoppages, "
                  f"H share of live {np.mean(x[np.isin(x, ['H', 'A'])] == 'H'):4.0%}")
        if k != first:
            g = (p != "X") & (P[first] != "X")
            print(f"{'':10s} agrees with {first}: ball seen {np.mean(p[g & seen] == P[first][g & seen]):.3f}, "
                  f"ball unseen {np.mean(p[g & ~seen] == P[first][g & ~seen]):.3f}; "
                  f"dead where the ball is unseen {np.mean(p[g & ~seen] == 'D'):.2f}, where seen {np.mean(p[g & seen] == 'D'):.2f}")
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")).label.values
    fr = np.flatnonzero(gt != "X")
    a, b = fr[0], fr[-1] + 1
    t = pm.label_codes(gt)
    print("\npublic 5 minutes (a check, 17 possessions): spellF1 with a bootstrap interval over possessions")
    for k, p in P.items():
        r = pm.evaluate(t, pm.label_codes(p), bootstrap_n=2000)
        ci = [v for kk, v in r.items() if "spell" in kk and ("ci" in kk or "lo" in kk or "hi" in kk)]
        print(f"  {k:10s} spellF1 {r['spell_f1_mean']:.3f} {ci}  macroF1 {r['macro_f1']:.3f}")
    print("\neach true possession or stoppage: best overlap (IoU) with a same-label stretch of each file, start and end offsets in s")
    s, e = runs(gt[a:b])
    for i, j in zip(s + a, e + a):
        line = f"  {gt[i]} {i:6d} {(j - i) / 25:5.1f} s |"
        for k, p in P.items():
            ps, pe = runs(p[a - 250:b + 250])
            best = (0, 0, 0)
            for u, v in zip(ps + a - 250, pe + a - 250):
                if p[u] != gt[i] or v <= i or u >= j:
                    continue
                iou = (min(j, v) - max(i, u)) / (max(j, v) - min(i, u))
                if iou > best[0]:
                    best = (iou, (u - i) / 25, (v - j) / 25)
            line += f" {k}: {best[0]:.2f} ({best[1]:+.1f}, {best[2]:+.1f}) |" if best[0] else f" {k}: missed |"
        print(line)


if __name__ == "__main__":
    main()
