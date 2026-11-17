"""Long-stoppage guard: where the plain all-data model calls a long dead ball, keep it dead.

Found by comparing our submission with the files other entries publish: in one 50 s stretch of
the second half (frames 168908-170168) the submission says Home in possession while the plain
all-data model, 441 of 445 published files and the tracking itself (ball lost at the touchline,
players walking at about 0.5 m/s for 40 s) say dead ball. The Tim-input model holds the
possession through it and merges two Home possessions around it.

The rule is code, not a hand edit: inside every run of `D` of at least --min-s seconds in the plain
all-data file (`alldata/learned/run.py --tags pwN pwNs1`), a possession run of the submission
lasting at least --min-alive-s seconds becomes `D`; shorter overlaps (stoppage edges) are left
alone. On BAR-ATH with the defaults (30 s, 10 s) it changes exactly that one stretch.

    python alldata/entries/stopfix.py --sub <submission.csv> --plain <alldata_learned.csv> --out <file>
"""

import argparse

import numpy as np
import pandas as pd


def long_runs(lab, value, min_len):
    m = lab == value
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(r[0], r[-1] + 1) for r in runs if len(r) >= min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-s", type=float, default=30.0, help="shortest plain dead run that counts")
    ap.add_argument("--min-alive-s", type=float, default=10.0,
                    help="only overwrite submission possession runs inside it at least this long (edges stay)")
    a = ap.parse_args()
    sub = pd.read_csv(a.sub)
    plain = pd.read_csv(a.plain)
    assert (sub.frame.values == plain.frame.values).all()
    lab = sub.label.values.copy()
    changed = 0
    for s, e in long_runs(plain.label.values, "D", int(a.min_s * 25)):
        seg = lab[s:e]
        alive = np.isin(seg, ["H", "A"]).astype(np.int8)
        for u, v in long_runs(alive, 1, int(a.min_alive_s * 25)):
            print(f"plain dead {sub.frame.values[s]}-{sub.frame.values[e - 1]} ({(e - s) / 25:.1f} s): "
                  f"submission {seg[u]} for {(v - u) / 25:.1f} s from frame {sub.frame.values[s + u]} -> D")
            seg[u:v] = "D"
            changed += v - u
    sub["label"] = lab
    sub.to_csv(a.out, index=False)
    print(f"{changed} frames changed -> {a.out}")


if __name__ == "__main__":
    main()
