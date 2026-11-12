"""External labels folded towards the challenge's controlled-possession rule.

The challenge changes possession only when the other team gains *controlled* possession: a touch
without control, a deflection or an aerial duel won without control is not a change (CHALLENGE.md,
DATA_POINTS.md). None of the external label sources uses that rule (README.md, "What the labels
are, and are not"), and DFL's in particular flips on a first touch, so 20-30% of their possessions
are under 2 s.

fold_uncontrolled() gives such a spell back to the team that had the ball. A spell is folded when
all of these hold:
- the other team had the ball just before it,
- after it, the other team has the ball again or the ball goes dead,
- it lasts under `seconds`,
- the tracking shows fewer than `min_touches` touches by its own team.

The two numbers come from the control test in DATA_POINTS.md, not from any ground truth: a team
has control when "the same team touches it twice" (MIN_TOUCHES = 2) or stays nearest "for about
2 s" (CONTROL_SECONDS = 2). It is a heuristic, not a re-labelling, so treat a score against the
folded labels as a second view, not the truth. Checked afterwards, not chosen on: the folded
labels have a median possession of 6.5-9.3 s and 4-10% under 2 s, in line with the public
window's shape. The v7 decoder's gain over v6 does not hinge on the 2 s. Its classifier was
trained on 2 s folds, and scored against labels folded at 1.5 / 2 / 3 s it gains
IDSSE +0.096 / +0.099 / +0.097, Metrica +0.026 / +0.039 / +0.040 and
SkillCorner +0.023 / +0.024 / +0.027 (external/score_control.py --fold-seconds, weight 0.15).

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
