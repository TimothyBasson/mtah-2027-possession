"""BAR-ATH's Smart Tagging set pieces as a clue for stoppages the players-only model plays through.

A tagged set piece (goal kick, corner, free kick, throw-in) says play stopped just before it. The
tag opens a few seconds before the restart (median 5 s before the model's own restart, 59 tags
against about 100 stoppages), so it cannot time anything; it is used only where the model shows
no stoppage at all near a tag. There the dead-ball odds are multiplied by ODDS over the stretch
leading up to the tag and the half is decoded again, so the stoppage is drawn where the model
was already in two minds (typically a goalkeeper with the ball: goal kick or open play?).

Smart Tagging is an automatic product of the same video and tracking (FAQ.md), not ground truth
and not the ball. No outside match has it, so this cannot be scored held out: it is off unless
asked for (deep/run.py --tags).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep")]

ODDS = 4.0
NEAR = (-3.0, 6.0)     # seconds around a tag's start in which a decoded stoppage explains the tag
STRETCH = (-20.0, 6.0)  # seconds around the start over which the odds are raised when none does


def set_pieces():
    from starter import load_smart_tagging as st
    return sorted(x["start_frame"] for x in st.load()
                  if x["code"].lower() == "set pieces" and "kick off" not in x["tags"]["Type"].lower())


def apply(lp, path, a, starts, fps=25, odds=ODDS):
    """lp (n, 3) log-probabilities of a half starting at video frame a, path its decoded states.
    Returns lp with the dead-ball odds raised near unexplained tags, and those tags."""
    lp = lp.copy()
    hit = []
    for s in starts:
        i = s - a
        if not 0 <= i < len(lp):
            continue
        lo, hi = max(0, i + int(NEAR[0] * fps)), min(len(lp), i + int(NEAR[1] * fps))
        if (path[lo:hi] == 2).any():
            continue
        lo, hi = max(0, i + int(STRETCH[0] * fps)), min(len(lp), i + int(STRETCH[1] * fps))
        lp[lo:hi, 2] += np.log(odds)
        hit.append(s)
    return lp, hit
