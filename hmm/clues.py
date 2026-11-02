"""Per-frame clues from the raw ATD. v1 has one: the nearest-player distance gap.

    gap = d(nearest away player to the ball) - d(nearest home player), in metres

Positive means home is closer. The gap is binned into five categories, and a frame where the
tracking has no ball gets a sixth, BALL_MISSING -- that is evidence too (soft evidence for D),
not a hole. A frame with a ball but no tracked player on one side has no gap and is -1, which the
emission model treats as "nothing observed".

The line format and the home/away slot mapping are the same as starter/baseline.py, which this
reuses; the difference is that we keep both distances instead of only the winner.
"""

import os

import numpy as np

from starter.baseline import FIELD_SEP, GROUP_SEP, HOME, MISSING

# Bin edges on the gap, in metres. Right-inclusive: (-inf, -3], (-3, -1], (-1, 1], (1, 3], (3, inf).
GAP_EDGES = np.array([-3.0, -1.0, 1.0, 3.0])
N_GAP_BINS = len(GAP_EDGES) + 1
BALL_MISSING = N_GAP_BINS
N_CATEGORIES = N_GAP_BINS + 1


def _xy(group, length, width):
    comma = group.index(",")
    return (float(group[:comma]) * length,
            float(group[comma + 1:group.index(",", comma + 1)]) * width)


def line_distances(line, team_of_track, pitch):
    """(video_frame, ball_seen, d_home, d_away) for one raw ATD line. NaN where nobody is seen."""
    length, width = pitch
    parts = line.split(FIELD_SEP)
    frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
    nearest = [np.inf, np.inf]
    ball = parts[2]
    if ball[0] == MISSING:
        return frame, False, np.nan, np.nan

    ball_x, ball_y = _xy(ball, length, width)
    for slot, group in enumerate(parts[1].split(GROUP_SEP)):
        if group[0] == MISSING:
            continue
        x, y = _xy(group, length, width)
        distance = (x - ball_x) ** 2 + (y - ball_y) ** 2
        side = 0 if team_of_track[slot] == HOME else 1
        if distance < nearest[side]:
            nearest[side] = distance
    d_home, d_away = (np.sqrt(d) if np.isfinite(d) else np.nan for d in nearest)
    return frame, True, d_home, d_away


def read_distances(raw_path, team_of_track, n_frames, pitch):
    """Stream the raw ATD once. Frames with no tracking line count as ball missing."""
    ball_seen = np.zeros(n_frames, dtype=bool)
    d_home = np.full(n_frames, np.nan)
    d_away = np.full(n_frames, np.nan)
    with open(raw_path) as handle:
        for line in handle:
            if line.count(FIELD_SEP) != 2:
                continue
            frame, seen, home, away = line_distances(line, team_of_track, pitch)
            ball_seen[frame], d_home[frame], d_away[frame] = seen, home, away
    return ball_seen, d_home, d_away


def gap_category(ball_seen, d_home, d_away):
    """Category per frame: 0..4 gap bins (away-much-closer .. home-much-closer), 5 ball missing,
    -1 no gap."""
    gap = d_away - d_home
    cats = np.digitize(np.nan_to_num(gap), GAP_EDGES, right=True).astype(np.int8)
    cats[np.isnan(gap)] = -1
    cats[~ball_seen] = BALL_MISSING
    return cats


def load_distances(raw_path, team_of_track, n_frames, pitch, cache_path):
    """read_distances, cached as .npz. The cache lives under data/, which is never committed."""
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached["ball_seen"]) == n_frames:
            return cached["ball_seen"], cached["d_home"], cached["d_away"]
    out = read_distances(raw_path, team_of_track, n_frames, pitch)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, ball_seen=out[0], d_home=out[1], d_away=out[2])
    return out
