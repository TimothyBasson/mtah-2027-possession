"""Goalkeepers on BAR-ATH's tracking get the team of the goal they stand in.

BAR-ATH's ATD (Metrica's automatic tracking) has anonymous track slots, each with a fixed team in
the metadata, and one slot per team marked as the goalkeeper (slots 23 and 45). Those two slots do
not stay on their own keeper. Measured over the two halves (199k frames, graded frames only):

- The away keeper slot stands within 25 m of the HOME goal on 16,392 frames (10.3% of graded
  frames), the home keeper slot within 25 m of the away goal on 1,554 (1.0%). The keeper of
  the slot's own team is then not in his own penalty area at all (own goal area empty on
  91-100% of those frames), and the slot stands 4-15 m from the other keeper slot: the tracker
  has carried the slot onto someone in the far goalmouth.
- 5,826 of those frames are a frozen track: the away keeper slot parked at one spot beside the
  home goal (x = 101.3 m, speed 0) for 215 s in the second half, and for 18 s in the first. The
  home keeper slot freezes too (31 s at midfield, 4 s in its box). No other slot ever freezes.

So the model saw an away "player" standing in the home goalmouth for minutes at a time; every
ball near that goal had an opponent beside it. No training match has this: their slots are named
players, and a keeper is within 25 m of the other goal only on the odd late corner (three halves,
about 20 s each).

fix() does two things, inside each graded half only:

1. A track that does not move at all for a second or more (x and y identical frame to frame) is
   unseen there. Real players are never perfectly still; on BAR-ATH only the keeper slots do this.
2. A keeper slot within KEEPER_ZONE of a goal line counts for the team defending that goal. It is
   moved, for those frames, into an extra slot of that team (one extra slot per team), so the team
   map stays one team per slot and every model reads it unchanged. Speed is recomputed.

This is the same rule Tim's hmm/clues.py applies to his own inputs (team_by_frame), which is why his
decoder already reads the keepers right; here it reaches every model that loads BAR-ATH through
possession/load.py. The training matches are left alone: their keepers are on the right team.
"""

import warnings
import xml.etree.ElementTree as ET

import numpy as np

KEEPER_ZONE = 25.0   # metres from a goal line: the penalty area plus a margin (as hmm/clues.py)
FROZEN_S = 1.0       # seconds without any change in position before a track counts as frozen


def keeper_slots(meta_path):
    """Slot indices the ATD metadata marks as goalkeepers (position_type "Goalkeeper")."""
    root = ET.parse(meta_path).getroot()
    out = []
    for slot, player in enumerate(root.iter("Player")):
        for p in player.iter("ProviderParameter"):
            if p.findtext("Name") == "position_type" and p.findtext("Value") == "Goalkeeper":
                out.append(slot)
    return tuple(out)


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def frozen(xy, fps, seconds=FROZEN_S):
    """(n, K) True on frames inside a run of at least `seconds` where the slot's x, y do not change
    at all (the first frame of the run included)."""
    same = np.zeros(xy.shape[:2], bool)
    same[1:] = (xy[1:] == xy[:-1]).all(-1)
    out = np.zeros_like(same)
    for k in range(xy.shape[1]):
        for s, e in zip(*_runs(same[:, k])):
            if e - s >= seconds * fps:
                out[max(s - 1, 0):e, k] = True
    return out


def home_defends_low_x(x, team, keepers):
    outfield = np.ones(len(team), bool)
    outfield[list(keepers)] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return bool(np.nanmean(x[:, outfield & (team == 0)]) < np.nanmean(x[:, outfield & (team == 1)]))


def fix(players, team, halves, keepers, length, fps, speed_fn, report=None):
    """(players (n, K + 2, 3), team (K + 2,)) with frozen tracks hidden and keeper slots near a goal
    moved to the defending team's extra slot. `report`, a dict, gets the frame counts."""
    team = np.asarray(team)
    n, K, _ = players.shape
    xy = np.concatenate([players[..., :2], np.full((n, 2, 2), np.nan, players.dtype)], 1)
    team2 = np.r_[team, [0, 1]].astype(team.dtype)
    extra = {0: K, 1: K + 1}
    counts = dict(frozen=0, moved=0)
    keepers = list(keepers)
    if sorted(team[keepers].tolist()) != [0, 1]:
        warnings.warn(f"expected one keeper slot per team, got {keepers}; keepers left as they are")
        keepers = []
    for a, b in halves:
        seg = xy[a:b + 1]                    # a view: edits land in xy
        fz = frozen(seg, fps)
        counts["frozen"] += int(fz.sum())
        seg[fz] = np.nan
        if not keepers:
            continue
        low = 0 if home_defends_low_x(seg[:, :K, 0], team, keepers) else 1   # team defending x = 0
        for k in keepers:
            x = seg[:, k, 0]
            with np.errstate(invalid="ignore"):
                defending = np.where(x < KEEPER_ZONE, low, np.where(x > length - KEEPER_ZONE, 1 - low, -1))
            wrong = (defending >= 0) & (defending != team[k])
            if not wrong.any():
                continue
            to = extra[1 - team[k]]
            seg[wrong, to] = seg[wrong, k]
            seg[wrong, k] = np.nan
            counts["moved"] += int(wrong.sum())
    if report is not None:
        report.update(counts)
    out = np.concatenate([xy, speed_fn(xy, fps)[..., None]], -1).astype(players.dtype)
    return out, team2
