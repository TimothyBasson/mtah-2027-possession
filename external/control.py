"""External labels folded towards the challenge's controlled-possession rule.

The challenge changes possession only when the other team gains *controlled* possession: a touch
without control, a deflection or an aerial duel won without control is not a change (CHALLENGE.md,
DATA_POINTS.md). None of the external label sources uses that rule (README.md, "What the labels
are, and are not"), and DFL's in particular flips on a first touch. On the raw labels 20-30% of
possessions are under 2 s, against 2 of 17 in the public window.

fold_uncontrolled() gives such a spell back to the team that had the ball. A spell is folded when
all of these hold:
- the other team had the ball just before it,
- after it, the other team has the ball again or the ball goes dead,
- it lasts under `seconds`,
- the tracking shows fewer than `min_touches` touches by its own team.

That is a rough version of the control test in DATA_POINTS.md ("the same team touches it twice";
one team nearest for about 2 s). It is a heuristic, not a re-labelling: the 2 s default was chosen
so the folded labels look like the public window (median 6.5-9.3 s, 4-10% of possessions under
2 s), so treat a score against them as a second view, not the truth.

Touches come from hmm.ball.detect_touches on the ball *as recorded*, never on the ball the model
is shown (score_hmm.py --seen knocks it out), so the yardstick does not move with --seen.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hmm.ball import detect_touches  # noqa: E402

CONTROL_SECONDS = 2.0
MIN_TOUCHES = 2
# A touch this many seconds before a spell's first frame still counts as the spell's: the label
# often flips a few frames after the touch that starts it.
TOUCH_LEAD_SECONDS = 0.2


def fold_uncontrolled(labels, touch_frames, touch_teams, fps, seconds=CONTROL_SECONDS,
                      min_touches=MIN_TOUCHES):
    """Copy of labels (H/A/D/X per frame) with spells that fail the control test given back to the
    team that had the ball before them. touch_teams is 0 home, 1 away, -1 unassigned per touch.
    Repeats until nothing changes, since folding one spell can join two others."""
    out = np.array(labels, copy=True)
    touch_frames = np.asarray(touch_frames, dtype=int)
    touch_teams = np.asarray(touch_teams)
    lead = int(round(TOUCH_LEAD_SECONDS * fps))
    while True:
        starts = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
        stops = np.r_[starts[1:], len(out)]
        changed = False
        for i in range(1, len(starts) - 1):
            a, b = starts[i], stops[i]
            k = out[a]
            if k not in ("H", "A"):
                continue
            other = "A" if k == "H" else "H"
            if out[a - 1] != other or out[b] not in (other, "D"):
                continue
            team = 0 if k == "H" else 1
            own = np.sum((touch_frames >= a - lead) & (touch_frames < b) & (touch_teams == team))
            if b - a < seconds * fps and own < min_touches:
                out[a:b] = other
                changed = True
        if not changed:
            return out


def control_labels(d, seconds=CONTROL_SECONDS, min_touches=MIN_TOUCHES):
    """fold_uncontrolled() for one external match, as loaded from data/external/*.npz."""
    fps = int(d["fps"])
    ball = d["ball"].astype(float)
    players = d["players"].astype(float)
    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        d_home = np.nanmin(dist[:, d["team"] == 0], axis=1)
        d_away = np.nanmin(dist[:, d["team"] == 1], axis=1)
    touches = detect_touches(ball[:, 0], ball[:, 1], d_home, d_away, fps)
    return fold_uncontrolled(d["labels"], touches["frame"], touches["team"], fps, seconds,
                             min_touches)
