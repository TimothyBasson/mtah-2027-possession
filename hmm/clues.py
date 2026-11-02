"""Per-frame clues from the raw ATD. v1 has one: the nearest-player distance gap.

    gap = d(nearest away player to the ball) - d(nearest home player), in metres

Positive means home is closer. The gap is binned into five categories, and a frame where the
tracking has no ball gets a sixth, BALL_MISSING -- that is evidence too (soft evidence for D),
not a hole. A frame with a ball but no tracked player on one side has no gap and is -1, which the
emission model treats as "nothing observed".

The line format and the home/away slot mapping are the same as starter/baseline.py, which this
reuses; the difference is that we keep both distances instead of only the winner.
"""

import hashlib
import os

import numpy as np

from starter.baseline import FIELD_SEP, GROUP_SEP, HOME, MISSING

# Bump when the parsing below changes, so an old cache is never read back as new.
CLUES_VERSION = 1

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
    """Stream the raw ATD once. Frames with no tracking line count as ball missing.

    Returns (ball_seen, d_home, d_away, stats), where stats is (lines read, malformed lines
    skipped, frames with no line) -- both of the latter push towards D, so they are reported.
    A frame outside 0..n_frames-1 is an error, not something to index around.
    """
    ball_seen = np.zeros(n_frames, dtype=bool)
    d_home = np.full(n_frames, np.nan)
    d_away = np.full(n_frames, np.nan)
    has_line = np.zeros(n_frames, dtype=bool)
    n_lines = n_malformed = 0
    with open(raw_path) as handle:
        for line in handle:
            n_lines += 1
            if line.count(FIELD_SEP) != 2:
                n_malformed += 1
                continue
            frame, seen, home, away = line_distances(line, team_of_track, pitch)
            if not 0 <= frame < n_frames:
                raise ValueError("{}: frame_id {} is outside 1..{}".format(
                    raw_path, frame + 1, n_frames))
            ball_seen[frame], d_home[frame], d_away[frame] = seen, home, away
            has_line[frame] = True
    return ball_seen, d_home, d_away, (n_lines, n_malformed, int(np.sum(~has_line)))


def gap_category(ball_seen, d_home, d_away):
    """Category per frame: 0..4 gap bins (away-much-closer .. home-much-closer), 5 ball missing,
    -1 no gap."""
    gap = d_away - d_home
    cats = np.digitize(np.nan_to_num(gap), GAP_EDGES, right=True).astype(np.int8)
    cats[np.isnan(gap)] = -1
    cats[~ball_seen] = BALL_MISSING
    return cats


def cache_key(raw_path, team_of_track, n_frames, pitch):
    """Short hash of everything the distances depend on: parser version, the raw file (by size
    and modification time), the home/away slot map, the frame count and the pitch."""
    stat = os.stat(raw_path)
    digest = hashlib.sha1(repr((CLUES_VERSION, os.path.basename(raw_path), stat.st_size,
                                int(stat.st_mtime), n_frames, tuple(pitch))).encode())
    digest.update(np.asarray(team_of_track, dtype=np.int8).tobytes())
    return digest.hexdigest()[:12]


def load_distances(raw_path, team_of_track, n_frames, pitch, cache_dir):
    """read_distances, cached as .npz under cache_dir (data/, never committed).

    The file name carries cache_key(), so changing the parser, the data, the slot map or the pitch
    writes a new cache instead of silently reading a stale one.
    """
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "hmm_distances_{}.npz".format(
            cache_key(raw_path, team_of_track, n_frames, pitch)))
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (cached["ball_seen"], cached["d_home"], cached["d_away"],
                    tuple(int(x) for x in cached["stats"]))
    ball_seen, d_home, d_away, stats = read_distances(raw_path, team_of_track, n_frames, pitch)
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, ball_seen=ball_seen, d_home=d_home, d_away=d_away,
                            stats=np.array(stats))
    return ball_seen, d_home, d_away, stats
