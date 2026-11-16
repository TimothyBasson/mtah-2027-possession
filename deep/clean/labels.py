"""Training labels cleaned with the ball each outside match recorded. Training only: the scoring
labels are never touched (deep/fixed.py, deep/paired.py and deep/decode.py read data/external
themselves), and nothing here is used when a match is labelled.

    retime   every H<->A change is moved to the nearest touch by the team that gains the ball, when
             there is one within `window` seconds and the move does not swallow a neighbouring spell
    mask     live frames where the ball contradicts the label are left out of the loss (set to X):
             the ball is within 1.5 m of a player, a metre nearer the other team, for 0.6 s or more
    fold     spells under 2 s with fewer than two touches by their team go back to the team that had
             the ball (external/control.py, the folded yardstick's rule, here on training labels)
    lastmask live frames where the last touch was by the other team are left out of the loss, for at
             most `hold` seconds after that touch

Touches are hmm.ball.detect_touches on the recorded ball: a ball-touch detector, used here as
external/control.py uses it, not a possession model.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]
from hmm.ball import detect_touches  # noqa: E402


def ball_view(d):
    """Touches and each team's nearest distance to the ball, per frame."""
    ball, players = d["ball"].astype(float), d["players"].astype(float)
    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dh, da = np.nanmin(dist[:, d["team"] == 0], 1), np.nanmin(dist[:, d["team"] == 1], 1)
    return detect_touches(ball[:, 0], ball[:, 1], dh, da, int(d["fps"])), dh, da


def retime(labels, touches, fps, window=1.5):
    """Copy of labels with H<->A changes moved onto the gaining team's nearest touch."""
    out = np.array(labels, copy=True)
    edge = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.r_[0, edge, len(out)]
    moved = 0
    for i, f in enumerate(edge):
        old, new = labels[f - 1], labels[f]
        if old not in ("H", "A") or new not in ("H", "A"):
            continue
        tf = touches["frame"][touches["team"] == (0 if new == "H" else 1)]
        if len(tf) == 0:
            continue
        k = int(tf[np.argmin(np.abs(tf - f))])
        lo, hi = bounds[i] + 1, bounds[i + 2] - 1  # keep a frame of each neighbour
        if k == f or abs(k - f) > window * fps or not lo <= k <= hi:
            continue
        if k < f:
            out[k:f] = new
        else:
            out[f:k] = old
        moved += 1
    return out, moved


def contradicted(labels, dh, da, fps, reach=1.5, margin=1.0, seconds=0.6):
    """Frames where one team is clearly on the ball for `seconds` or more and the label names the other."""
    live = (labels == "H") | (labels == "A")
    with np.errstate(invalid="ignore"):
        clear = live & (np.minimum(dh, da) < reach) & (np.abs(dh - da) > margin)
    who = np.where(da < dh, "A", "H")
    bad = clear & (who != labels)
    code = np.where(bad, who, "-")
    e = np.flatnonzero(code[1:] != code[:-1]) + 1
    out = np.zeros(len(labels), bool)
    for a, b in zip(np.r_[0, e], np.r_[e, len(labels)]):
        if bad[a] and b - a >= seconds * fps:
            out[a:b] = True
    return out


def last_touch_against(labels, touches, fps, hold=2.0):
    """Live frames within `hold` seconds after a touch by the team the label does not name, with no
    touch by the labelled team since."""
    n = len(labels)
    last = np.full(n, -1)
    age = np.full(n, 10 ** 9)
    ok = touches["team"] >= 0
    tf, tt = touches["frame"][ok], touches["team"][ok]
    idx = np.searchsorted(tf, np.arange(n), side="right") - 1
    has = idx >= 0
    last[has] = tt[idx[has]]
    age[has] = np.arange(n)[has] - tf[idx[has]]
    y = np.where(labels == "H", 0, np.where(labels == "A", 1, -1))
    return (y >= 0) & (last >= 0) & (last != y) & (age <= hold * fps)


def clean(d, mode):
    """Labels of one outside match for training under `mode`: a '+'-joined list of the above."""
    lab = np.array(d["labels"], copy=True)
    fps = int(d["fps"])
    touches, dh, da = ball_view(d)
    for step in mode.split("+"):
        if step == "retime":
            lab, _ = retime(lab, touches, fps)
        elif step == "mask":
            lab[contradicted(lab, dh, da, fps)] = "X"
        elif step == "fold":  # spells that fail the controlled-possession test go back (external/control.py)
            sys.path.insert(0, os.path.join(ROOT, "external"))
            from control import fold_uncontrolled
            lab = fold_uncontrolled(lab, touches["frame"], touches["team"], fps)
        elif step == "lastmask":
            lab[last_touch_against(lab, touches, fps)] = "X"
        else:
            raise ValueError(step)
    return lab


if __name__ == "__main__":
    import glob
    for prov in ("idsse", "metrica", "skillcorner"):
        rows = []
        for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", prov + "*.npz"))):
            d = np.load(p)
            lab, fps = d["labels"], int(d["fps"])
            t, dh, da = ball_view(d)
            live = (lab == "H") | (lab == "A")
            flips = (live[1:] & live[:-1] & (lab[1:] != lab[:-1])).sum()
            r, moved = retime(lab, t, fps)
            dead_min = (lab == "D").sum() / fps / 60
            rows.append([moved / max(1, flips), (r != lab).sum() / live.sum(),
                         contradicted(lab, dh, da, fps).sum() / live.sum(),
                         contradicted(r, dh, da, fps).sum() / live.sum(),
                         last_touch_against(lab, t, fps).sum() / live.sum(),
                         last_touch_against(r, t, fps).sum() / live.sum(),
                         (lab[t["frame"]] == "D").sum() / max(dead_min, 1e-9),
                         live[t["frame"]].sum() / (live.sum() / fps / 60)])
        m = np.mean(rows, 0)
        print(f"{prov:12s} turnovers moved {m[0]:.2f}  live frames relabelled {m[1]:.3f}  contradicted {m[2]:.3f} "
              f"(after retime {m[3]:.3f})  last touch against {m[4]:.3f} (after retime {m[5]:.3f})  "
              f"touches per dead minute {m[6]:.1f}, per live minute {m[7]:.1f}")
