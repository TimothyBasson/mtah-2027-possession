"""Do the teams react when the ball changes team, and how fast? The premise, measured.

    python alldata/react/premise.py

At every true turnover in open play (control-rule labels: a possession that starts straight
after the other team's), by how long the new possession lasts: the two teams' movement over the
next 1, 2 and 4 s, counted in the winner's direction of attack.
  both   sum of the two teams' mean advance (m/s): positive when both move the winner's way
  open   the winner's change of spread minus the loser's (m/s): positive when the winner opens
         and the loser closes
`share` is how often the number is positive (0.5 = no reaction to see).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rdata  # noqa: E402
import react as R  # noqa: E402
from common import PROVS, pm  # noqa: E402

BINS = [(0, 2, "under 2 s"), (2, 5, "2 to 5 s"), (5, 10, "5 to 10 s"), (10, 1e9, "over 10 s")]
POS = {n: i for i, n in enumerate(R.NAMES)}


def main():
    ms = rdata.load_all()
    rows = {p: [] for p in PROVS}
    for m in ms:
        t = m.folded
        Rm = np.zeros((len(t), len(R.NAMES)), np.float32)
        have = np.zeros(len(t), bool)
        for r, fr in zip(m.R, m.fr):
            Rm[fr], have[fr] = r, True
        rows_at = np.flatnonzero(have)  # rows sit at each half's first frame + STEP * k
        for a, b, k in pm.possession_spells(t, pm.valid_runs(t)):
            if a == 0 or t[a - 1] != 1 - k:
                continue
            j = np.searchsorted(rows_at, a)  # first row at or after the start
            if j == len(rows_at) or rows_at[j] - a >= R.STEP:
                continue
            f = rows_at[j]
            sgn = 1.0 if k == 0 else -1.0
            w, l = ("h", "a") if k == 0 else ("a", "h")
            x = Rm[f]
            rows[m.fold].append([(b - a) / 25] + [sgn * x[POS[f"both_{h}"]] for h in ("p1", "f1", "f2", "f4")] +
                                [x[POS[f"{w}_{h}_spr"]] - x[POS[f"{l}_{h}_spr"]] for h in ("p1", "f1", "f2", "f4")])
    heads = ["both -1 s", "both +1 s", "both +2 s", "both +4 s", "open -1 s", "open +1 s", "open +2 s", "open +4 s"]
    for p in PROVS:
        X = np.array(rows[p])
        print(f"\n{p}: mean (share positive) at the start of a possession won in open play")
        print(f"{'lasts':12s}{'n':>6s}" + "".join(f"{h:>16s}" for h in heads))
        for lo, hi, nm in BINS:
            k = (X[:, 0] >= lo) & (X[:, 0] < hi)
            print(f"{nm:12s}{k.sum():6d}" + "".join(f"{X[k, j].mean():+9.2f} ({(X[k, j] > 0).mean():.2f})" for j in range(1, 9)))


if __name__ == "__main__":
    main()
