"""Check the touch detector (hmm/ball.py) on the out-of-sample matches. Prints aggregates only.

    python external/touches.py [--dv 3 4 5] [--radius 2.5] [--tol 10]

Metrica (3 games) has event data, so the detector is checked against it directly:
    kicks       PASS and SHOT starts: share with a detected touch within --tol frames (recall), and
                of those, the share whose touch went to the event's team
    receptions  ends of completed PASSes: recall as above
    precision   detected touches within --tol frames of any event's start or end. Metrica does not
                log every touch (dribbles, a first touch before a pass), so this is a lower bound.
Only events with the ball measured within --tol frames are scored; the rest are the "unmeasurable"
share, which is about the tracking, not the detector.

IDSSE (7 games) has no events, only DFL's per-frame possession flag, so there:
    team agreement  of assigned touches with H / A both 0.4 s after the touch, the share where
                    that label is the touch's team
    switch recall   H<->A changes in the flag with an assigned touch by the new team within --tol
Metrica gets the team agreement too, against its event-rebuilt labels.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "external"))
from hmm.ball import TOUCH_DV, TOUCH_RADIUS, detect_touches  # noqa: E402
from score_hmm import nearest_by_team  # noqa: E402

EXT = os.path.join(ROOT, "data", "external")
METRICA = os.path.join(EXT, "raw", "metrica", "sample-data", "data")
AFTER = 10  # frames after a touch at which the possession label is read (0.4 s)


def metrica_events(game):
    """(team 0/1, type, start index, end index, completed) per event; index = frame - 1."""
    d = os.path.join(METRICA, f"Sample_Game_{game}")
    if game < 3:
        ev = pd.read_csv(os.path.join(d, f"Sample_Game_{game}_RawEventsData.csv"))
        return [(0 if r.Team == "Home" else 1, r.Type, int(r["Start Frame"]) - 1,
                 int(r["End Frame"]) - 1, isinstance(r.To, str)) for _, r in ev.iterrows()]
    raw = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]
    home = "FIFATMA"  # the first <Team> in Sample_Game_3_metadata.xml, as kloppy reads it
    return [(0 if r["team"]["id"] == home else 1, r["type"]["name"], r["start"]["frame"] - 1,
             r["end"]["frame"] - 1, bool(r.get("to"))) for r in raw]


def nearest_offset(targets, frames):
    """Per target frame, the signed offset to the nearest of the sorted frames (inf if none)."""
    frames = np.asarray(frames)
    if not len(frames):
        return np.full(len(targets), np.inf)
    i = np.clip(np.searchsorted(frames, targets), 1, len(frames) - 1) if len(frames) > 1 else \
        np.zeros(len(targets), dtype=int)
    left = frames[np.maximum(i - 1, 0)] - targets
    right = frames[i] - targets
    return np.where(np.abs(left) <= np.abs(right), left, right)


def graded(d):
    keep = np.zeros(len(d["labels"]), dtype=bool)
    for a, b in d["halves"]:
        keep[a:b + 1] = True
    return keep


def run(d, threshold, radius):
    d_home, d_away = nearest_by_team(d["players"], d["ball"], d["team"])
    t = detect_touches(d["ball"][:, 0], d["ball"][:, 1], d_home, d_away, int(d["fps"]), threshold,
                       radius)
    keep = graded(d)[t["frame"]]
    return {k: (v[keep] if k in ("frame", "team", "dist", "dv", "speed_before", "speed_after")
                else v) for k, v in t.items()}


def team_agreement(t, labels):
    after = np.clip(t["frame"] + AFTER, 0, len(labels) - 1)
    lab = labels[after]
    ok = (t["team"] >= 0) & np.isin(lab, ["H", "A"])
    return np.sum(np.array(["H", "A"])[t["team"][ok]] == lab[ok]), np.sum(ok)


def measurable(dv, frames, tol):
    """Events with the velocity change measured somewhere within tol frames."""
    fin = np.isfinite(dv)
    return np.array([fin[max(f - tol, 0):f + tol + 1].any() for f in frames], dtype=bool)


def metrica(threshold, radius, tol):
    rows = []
    for g in (1, 2, 3):
        d = np.load(os.path.join(EXT, f"metrica_{g}.npz"))
        t = run(d, threshold, radius)
        ev = [e for e in metrica_events(g) if e[1] != "CARD"]
        kicks = np.array([(s, side) for side, kind, s, _, _ in ev if kind in ("PASS", "SHOT")])
        recs = np.array([e for side, kind, _, e, done in ev if kind == "PASS" and done])
        every = np.unique([f for e in ev for f in e[2:4]])
        r = {"minutes": graded(d).sum() / d["fps"] / 60, "touches": len(t["frame"]),
             "assigned": np.mean(t["team"] >= 0)}
        for name, frames in (("kick", kicks[:, 0]), ("rec", recs)):
            m = measurable(t["dv_frames"], frames, tol)
            off = nearest_offset(frames[m], t["frame"])
            r[name + "_measurable"] = m.mean()
            r[name + "_recall"] = np.mean(np.abs(off) <= tol)
            if name == "kick":
                hit = np.abs(off) <= tol
                idx = np.searchsorted(t["frame"], frames[m][hit] + off[hit])
                side = t["team"][idx]
                r["kick_unassigned"] = np.mean(side < 0)
                r["kick_team"] = np.mean(side[side >= 0] == kicks[m][hit][side >= 0, 1])
                r["kick_offset_median"] = np.median(np.abs(off[hit]))
        r["precision"] = np.mean(np.abs(nearest_offset(t["frame"], every)) <= tol)
        agree, n = team_agreement(t, d["labels"])
        r["label_agree"] = agree / n
        rows.append(r)
    return rows


def idsse(threshold, radius, tol):
    rows = []
    for path in sorted(glob.glob(os.path.join(EXT, "idsse_*.npz"))):
        d = np.load(path)
        t = run(d, threshold, radius)
        labels = d["labels"]
        agree, n = team_agreement(t, labels)
        prev, nxt = labels[:-1], labels[1:]
        alive = np.isin(prev, ["H", "A"]) & np.isin(nxt, ["H", "A"])
        switch = np.flatnonzero(alive & (prev != nxt))
        new_team = (labels[switch + 1] == "A").astype(int)
        found = 0
        for f, side in zip(switch, new_team):
            near = np.abs(t["frame"] - f) <= tol
            found += np.any(near & (t["team"] == side))
        rows.append({"minutes": graded(d).sum() / d["fps"] / 60, "touches": len(t["frame"]),
                     "assigned": np.mean(t["team"] >= 0), "label_agree": agree / n,
                     "switch_recall": found / max(len(switch), 1)})
    return rows


def show(name, rows):
    keys = list(rows[0])
    mean = {k: np.mean([r[k] for r in rows]) for k in keys}
    mean["touches_per_min"] = np.mean([r["touches"] / r["minutes"] for r in rows])
    print(f"  {name}: " + "  ".join(f"{k} {v:.3f}" if v < 10 else f"{k} {v:.0f}"
                                   for k, v in mean.items() if k not in ("minutes", "touches")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dv", type=float, nargs="+", default=[TOUCH_DV])
    ap.add_argument("--radius", type=float, default=TOUCH_RADIUS)
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--skip-idsse", action="store_true")
    a = ap.parse_args()
    for dv in a.dv:
        print(f"dv >= {dv} m/s, radius {a.radius} m, tolerance {a.tol} frames")
        show("metrica", metrica(dv, a.radius, a.tol))
        if not a.skip_idsse:
            show("idsse  ", idsse(dv, a.radius, a.tol))
