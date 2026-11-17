"""Boundary-shift test from other entries (team-switch delay / edge shift).

Several public entries move every predicted hand-over by a fixed lag ("team-switch delay",
"DFL-estimated edge shift"). This decodes the saved held-out runs of a tag exactly as score.py
does, then moves each H<->A boundary (and optionally each live<->dead boundary) by k frames
at 25 fps, and scores held out by provider, as built and folded.

    python alldata/entries/shift.py --tag pw1e --key s
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "alldata", "learned"))
import score as S  # noqa: E402

SHIFTS = (-12, -6, -3, 3, 6, 12)


def shift_edges(lab, k, which):
    """Move boundaries between runs by k frames (k > 0: later). which: 'team' (H<->A only),
    'dead' (any boundary touching D), 'all'."""
    if k == 0:
        return lab
    out = lab.copy()
    edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for e in edges:
        a, b = lab[e - 1], lab[e]
        if "X" in (a, b):
            continue
        team = {a, b} == {"H", "A"}
        if (which == "team" and not team) or (which == "dead" and team):
            continue
        if k > 0:  # old label extends into the new run
            hi = e + k
            seg = slice(e, min(hi, len(lab)))
            m = lab[seg] == b
            out[seg][m] = a
        else:
            lo = max(e + k, 0)
            seg = slice(lo, e)
            m = lab[seg] == a
            out[seg][m] = b
    return out


def one(args):
    name, f, key = args
    raw, fold, spans = S.truths(name)
    z = np.load(f)
    res = {}
    for t, dec, pre in ((raw, S.RAW, ""), (fold, S.FOLDED, "f")):
        kk = pre + key if f"{pre}{key}0" in z.files else key
        probs = [z[f"{kk}{i}"] for i in range(len(spans))]
        lab = S.decode.labels_from(probs, spans, len(t), **dec)
        for which in ("team", "dead"):
            for k in (0,) + SHIFTS:
                if k == 0 and which == "dead":
                    continue
                r = S.pm.evaluate(t, S.pm.label_codes(shift_edges(lab, k, which)), bootstrap_n=0)
                res[(pre or "b", which if k else "-", k)] = r["spell_f1_mean"]
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--key", default="s")
    a = ap.parse_args()
    jobs = []
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(a.tag), fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                jobs.append((os.path.basename(f)[:-4], f, a.key))
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(one, jobs))
    provs = sorted({n.split("_")[0] for n, _ in out})
    keys = list(out[0][1])
    print(f"{a.tag}, key {a.key}: spellF1 by provider (mean of providers); k in frames at 25 fps, + = later")
    for key in keys:
        per = [np.mean([r[key] for n, r in out if n.startswith(p)]) for p in provs]
        base = (key[0], "-", 0)
        d = [r[key] - r[base] for _, r in out]
        print(f"  {key[0]:2s} {key[1]:5s} {key[2]:+4d}  " + " ".join(f"{p[:6]} {x:.3f}" for p, x in zip(provs, per))
              + f"   mean {np.mean(per):.4f}   d {np.mean(d):+.4f}")


if __name__ == "__main__":
    main()
