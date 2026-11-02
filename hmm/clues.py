"""Per-frame clues from the raw ATD. Two of them, each an int category per frame, -1 for "nothing
observed" (which the emission model skips):

1. The nearest-player gap, for who has the ball:

    gap = d(nearest away player to the ball) - d(nearest home player), in metres

   Positive means home is closer, binned into five categories. It is only observed on frames with
   a usable ball and a tracked player on each side.

2. Where the ball is, for whether it is in play. A ball seen clearly past a line is out of play.
   A missing ball is not evidence on its own -- the tracking loses it in open play all the time --
   but where it was last seen is: a ball lost at or past a line has probably gone out. That last
   sighting is carried across the missing stretch for a limited number of frames.

   A "ball" far outside the pitch is most likely a spare ball by the touchline or a false
   detection, so it is treated as no ball at all, for both clues.

The line format and the home/away slot mapping are the same as starter/baseline.py, which this
reuses; the difference is that we keep both distances instead of only the winner.
"""

import hashlib
import os

import numpy as np

from starter.baseline import FIELD_SEP, GROUP_SEP, HOME, MISSING

# Bump when the parsing below changes, so an old cache is never read back as new.
CLUES_VERSION = 2

# Bin edges on the gap, in metres. Right-inclusive: (-inf, -3], (-3, -1], (-1, 1], (1, 3], (3, inf).
GAP_EDGES = np.array([-3.0, -1.0, 1.0, 3.0])
N_GAP_BINS = len(GAP_EDGES) + 1

# Ball-place categories.
IN_PLAY = 0             # seen, on the pitch or within OUT_MARGIN of a line
OUTSIDE = 1             # seen more than OUT_MARGIN (and at most FAR_OUTSIDE) past a line
MISSING_AFTER_LINE = 2  # not seen; last seen within LINE_MARGIN inside a line, or past it
MISSING_AFTER_PLAY = 3  # not seen; last seen anywhere else
MISSING_STALE = 4       # not seen for longer than the carry, or never seen before
N_PLACES = 5

# Metres. The tracked point is the ball's centre and is noisy, and a ball on the line is still in
# play, so "outside" needs a margin. At Smart Tagging throw-ins the ball sits at a median 0.0 m
# from the touchline, and goal kicks at 4.5 m from the goal line, so the geometry holds.
OUT_MARGIN = 1.0
LINE_MARGIN = 2.0
FAR_OUTSIDE = 5.0


def _xy(group, length, width):
    comma = group.index(",")
    return (float(group[:comma]) * length,
            float(group[comma + 1:group.index(",", comma + 1)]) * width)


def line_distances(line, team_of_track, pitch):
    """(video_frame, ball_x, ball_y, d_home, d_away) for one raw ATD line, in metres.

    NaN where nobody, or no ball, is seen."""
    length, width = pitch
    parts = line.split(FIELD_SEP)
    frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
    nearest = [np.inf, np.inf]
    ball = parts[2]
    if ball[0] == MISSING:
        return frame, np.nan, np.nan, np.nan, np.nan

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
    return frame, ball_x, ball_y, d_home, d_away


def read_distances(raw_path, team_of_track, n_frames, pitch):
    """Stream the raw ATD once. Frames with no tracking line count as ball missing.

    Returns (ball_x, ball_y, d_home, d_away, stats) in metres, NaN where not seen, and stats is
    (lines read, malformed lines skipped, frames with no line), reported so that a broken file
    does not pass for a long stretch of missing ball.
    A frame outside 0..n_frames-1 is an error, not something to index around.
    """
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
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
            frame, x, y, home, away = line_distances(line, team_of_track, pitch)
            if not 0 <= frame < n_frames:
                raise ValueError("{}: frame_id {} is outside 1..{}".format(
                    raw_path, frame + 1, n_frames))
            ball_x[frame], ball_y[frame], d_home[frame], d_away[frame] = x, y, home, away
            has_line[frame] = True
    return ball_x, ball_y, d_home, d_away, (n_lines, n_malformed, int(np.sum(~has_line)))


def metres_outside(ball_x, ball_y, pitch):
    """Signed distance past the nearest line: positive outside the pitch, negative inside."""
    length, width = pitch
    return np.maximum.reduce([-ball_x, ball_x - length, -ball_y, ball_y - width])


def ball_is_usable(ball_x, ball_y, pitch):
    """Seen, and not so far outside that it is probably not the match ball."""
    with np.errstate(invalid="ignore"):
        return metres_outside(ball_x, ball_y, pitch) <= FAR_OUTSIDE


def gap_category(usable, d_home, d_away):
    """Category per frame: 0..4 gap bins (away-much-closer .. home-much-closer), -1 where there is
    no usable ball or no tracked player on one side."""
    gap = d_away - d_home
    cats = np.digitize(np.nan_to_num(gap), GAP_EDGES, right=True).astype(np.int8)
    cats[np.isnan(gap) | ~usable] = -1
    return cats


def ball_place_category(ball_x, ball_y, pitch, carry_frames):
    """Ball-place category per frame (IN_PLAY ... MISSING_STALE, see the top of this file).

    A missing frame takes its category from the last usable sighting before it, for at most
    carry_frames frames after that sighting; after that, or with no sighting yet, it is stale.
    """
    usable = ball_is_usable(ball_x, ball_y, pitch)
    with np.errstate(invalid="ignore"):
        outside = metres_outside(ball_x, ball_y, pitch)
    cats = np.where(usable & (outside > OUT_MARGIN), OUTSIDE, IN_PLAY).astype(np.int8)

    frames = np.arange(len(usable))
    last = np.maximum.accumulate(np.where(usable, frames, -1))  # last usable frame, -1 if none
    missing = ~usable
    known = missing & (last >= 0) & (frames - last <= carry_frames)
    at_line = np.zeros(len(usable), dtype=bool)
    at_line[known] = outside[last[known]] >= -LINE_MARGIN
    cats[missing] = MISSING_STALE
    cats[known & at_line] = MISSING_AFTER_LINE
    cats[known & ~at_line] = MISSING_AFTER_PLAY
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
            return (cached["ball_x"], cached["ball_y"], cached["d_home"], cached["d_away"],
                    tuple(int(x) for x in cached["stats"]))
    ball_x, ball_y, d_home, d_away, stats = read_distances(raw_path, team_of_track, n_frames, pitch)
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, ball_x=ball_x, ball_y=ball_y, d_home=d_home,
                            d_away=d_away, stats=np.array(stats))
    return ball_x, ball_y, d_home, d_away, stats
