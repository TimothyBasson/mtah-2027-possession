"""How far each provider's possession labels agree with the ball it recorded.

    python deep/clean/diag.py

Per provider, from the 30 outside matches (the ball is read here to judge labels, nothing more):
  seen       share of live (H/A) frames with the ball recorded
  near       of live frames with the ball within 1.5 m of a player and 1 m nearer one team than the
             other: share where that team is the labelled team
  held>=1s   the same over stretches where one team stays so for a second or more
  turnovers  H<->A changes per live minute
  touch      share of those with a touch by the new team within 0.25 s / 1 s
  offset     label change minus the nearest new-team touch within 3 s: median and quartiles, seconds
  short      share of possessions under 1 s / 2 s
"""

import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]
from hmm.ball import detect_touches  # noqa: E402  (a ball-touch detector, not a possession model)


def nearest(d):
    ball, players = d["ball"].astype(float), d["players"].astype(float)
    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(dist[:, d["team"] == 0], 1), np.nanmin(dist[:, d["team"] == 1], 1)


def runs(v):
    e = np.flatnonzero(v[1:] != v[:-1]) + 1
    return np.r_[0, e], np.r_[e, len(v)]


def one(path):
    d = np.load(path)
    fps = int(d["fps"])
    lab = d["labels"]
    dh, da = nearest(d)
    t = detect_touches(d["ball"][:, 0], d["ball"][:, 1], dh, da, fps)
    live = (lab == "H") | (lab == "A")
    seen = np.isfinite(d["ball"][:, 0])
    y = np.where(lab == "H", 0, 1)
    with np.errstate(invalid="ignore"):
        clear = live & seen & (np.minimum(dh, da) < 1.5) & (np.abs(dh - da) > 1.0)
    who = (da < dh).astype(int)
    agree = (who == y)[clear]
    # stretches of a second or more with one team clearly on the ball
    code = np.where(clear, who, -1)
    s, e = runs(code)
    keep = (code[s] >= 0) & (e - s >= fps)
    held = np.concatenate([(y[a:b] == code[a]) for a, b in zip(s[keep], e[keep])]) if keep.any() else np.zeros(0, bool)
    flips = np.flatnonzero(live[1:] & live[:-1] & (lab[1:] != lab[:-1])) + 1
    offs, hit25, hit100 = [], 0, 0
    for f in flips:
        new = 0 if lab[f] == "H" else 1
        tf = t["frame"][t["team"] == new]
        if len(tf) == 0:
            continue
        k = tf[np.argmin(np.abs(tf - f))]
        dt = (f - k) / fps
        hit25 += abs(dt) <= 0.25
        hit100 += abs(dt) <= 1.0
        if abs(dt) <= 3:
            offs.append(dt)
    s, e = runs(lab)
    poss = (e - s)[(lab[s] == "H") | (lab[s] == "A")] / fps
    return dict(seen=seen[live].mean(), near=agree.mean(), held=held.mean() if len(held) else np.nan,
                clear=clear.sum() / max(1, live.sum()), per_min=len(flips) / (live.sum() / fps / 60),
                hit25=hit25 / max(1, len(flips)), hit100=hit100 / max(1, len(flips)), offs=np.array(offs),
                short1=(poss < 1).mean(), short2=(poss < 2).mean(), median=np.median(poss))


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    for prov in ("idsse", "metrica", "skillcorner"):
        rs = [one(p) for p in paths if os.path.basename(p).startswith(prov)]
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        o = np.concatenate([r["offs"] for r in rs])
        q = np.percentile(o, [25, 50, 75])
        print(f"{prov:12s} seen {m('seen'):.2f}  clear {m('clear'):.2f}  near {m('near'):.3f}  held>=1s {m('held'):.3f}  "
              f"turnovers/min {m('per_min'):.1f}  touch 0.25s {m('hit25'):.2f} 1s {m('hit100'):.2f}  "
              f"offset {q[1]:+.2f} ({q[0]:+.2f}, {q[2]:+.2f})  under 1s {m('short1'):.2f} 2s {m('short2'):.2f}  median {m('median'):.1f}s")


if __name__ == "__main__":
    main()
