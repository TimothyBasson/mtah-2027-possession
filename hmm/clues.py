"""Per-frame clues from the raw ATD. Three of them, each an int category per frame, -1 for "nothing
observed" (which the emission model skips):

1. The nearest-player gap, for who has the ball:

    gap = d(nearest away player to the ball) - d(nearest home player), in metres

   Positive means home is closer, binned into five categories. It is only observed on frames with
   a usable ball and a tracked player on each side.

   The two goalkeeper slots are the exception to the metadata's home/away slot map. When one
   keeper drops out of tracking at the far end, the tracker often carries the near keeper under
   the absent keeper's slot -- the away keeper's slot moving with the home keeper in the home goal,
   say -- and the keeper on the ball then counts for the wrong team. So within each half a keeper
   slot takes the team of the end it stands in: the team defending that half of the pitch, found
   from where each team's outfield slots stand on average over the half.

2. Where the ball is, for whether it is in play. A ball seen clearly past a line is out of play.
   A missing ball is not evidence on its own -- the tracking loses it in open play all the time --
   but where it was last seen is: a ball lost at or past a line has probably gone out. That last
   sighting is carried across the missing stretch for a limited number of frames.

   A "ball" far outside the pitch is most likely a spare ball by the touchline or a false
   detection, so it is treated as no ball at all, for both clues.

3. How fast the players move, for whether the ball is in play. When it is dead, players walk to
   their places; in play most of them run. It is the median over the tracked players, so it
   does not need the ball at all, and so it also says what a missing ball means: lost while the
   players run, it is probably still in play; while they walk, probably dead.

The line format and the home/away slot mapping are the same as starter/baseline.py, which this
reuses; the difference is that we keep both distances instead of only the winner.
"""

import hashlib
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from starter.baseline import AWAY, FIELD_SEP, GROUP_SEP, HOME, MISSING

# Bump when the parsing below changes, so an old cache is never read back as new.
CLUES_VERSION = 5

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

# Player speed is measured over SPEED_LAG frames (0.2 s at 25 fps), as in external/build.py, from
# positions; the speed field in the ATD is not used. The window is centred on the frame, since the
# decoding is offline. A step faster than MAX_SPEED m/s is no footballer but a tracked identity
# jumping to another player, and is dropped. A median over fewer than MIN_PLAYERS measured players
# is too thin to say the team is walking, and counts as unmeasured.
SPEED_LAG = 5
MAX_SPEED = 12.0
MIN_PLAYERS = 6
# Bin edges on the median player speed, m/s: standing, walking, jogging, running.
SPEED_EDGES = np.array([0.6, 0.9, 1.2, 1.5, 2.0, 2.5])
N_SPEED_BINS = len(SPEED_EDGES) + 1

# Frames per block when computing distances over the whole match, to bound memory.
CHUNK = 20000


def _xy(group, length, width):
    comma = group.index(",")
    return (float(group[:comma]) * length,
            float(group[comma + 1:group.index(",", comma + 1)]) * width)


def keeper_slots(meta_path):
    """Slot indices the ATD metadata marks as goalkeepers (position_type "Goalkeeper")."""
    root = ET.parse(meta_path).getroot()
    slots = []
    for slot, player in enumerate(root.iter("Player")):
        for parameter in player.iter("ProviderParameter"):
            if (parameter.findtext("Name") == "position_type"
                    and parameter.findtext("Value") == "Goalkeeper"):
                slots.append(slot)
    return tuple(slots)


def home_defends_low_x(px, team_of_track, keepers):
    """True if the home team defends the x = 0 end over these frames: its outfield slots stand at a
    lower x than the away team's on the median frame."""
    outfield = np.ones(len(team_of_track), dtype=bool)
    outfield[list(keepers)] = False
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody of a team tracked
        mean_x = [np.nanmean(px[:, outfield & (team_of_track == side)], axis=1)
                  for side in (HOME, AWAY)]
        return bool(np.nanmedian(mean_x[0]) < np.nanmedian(mean_x[1]))


def team_by_frame(px, team_of_track, keepers, halves, length):
    """(frames, slots) team per slot per frame: the metadata's map, except that inside each half a
    keeper slot takes the team defending the half of the pitch it stands in (see the top of this
    file). Outside the halves, and wherever a keeper slot is untracked, the metadata's map holds."""
    team = np.tile(np.asarray(team_of_track, dtype=np.int8), (len(px), 1))
    for lo, hi in halves:
        low_end = HOME if home_defends_low_x(px[lo:hi], team_of_track, keepers) else AWAY
        for slot in keepers:
            x = px[lo:hi, slot]
            seen = np.isfinite(x)
            team[lo:hi, slot][seen] = np.where(x[seen] < length / 2, low_end, 1 - low_end)
    return team


def nearest_by_team(px, py, ball_x, ball_y, team):
    """(d_home, d_away): distance from the ball to the nearest tracked slot of each team per frame,
    metres, NaN where the ball or every slot of that team is unseen. team is (frames, slots)."""
    d_home = np.full(len(px), np.nan)
    d_away = np.full(len(px), np.nan)
    for lo in range(0, len(px), CHUNK):
        hi = lo + CHUNK
        d = np.hypot(px[lo:hi] - ball_x[lo:hi, None], py[lo:hi] - ball_y[lo:hi, None])
        d = np.where(np.isfinite(d), d, np.inf)
        for side, out in ((HOME, d_home), (AWAY, d_away)):
            nearest = np.min(np.where(team[lo:hi] == side, d, np.inf), axis=1)
            out[lo:hi] = np.where(np.isfinite(nearest), nearest, np.nan)
    return d_home, d_away


def line_positions(line, n_slots, pitch):
    """(video_frame, xs, ys) for one raw ATD line: every player slot in metres, NaN if untracked."""
    length, width = pitch
    parts = line.split(FIELD_SEP)
    xs = np.full(n_slots, np.nan)
    ys = np.full(n_slots, np.nan)
    for slot, group in enumerate(parts[1].split(GROUP_SEP)):
        if group[0] != MISSING:
            xs[slot], ys[slot] = _xy(group, length, width)
    return int(parts[0]) - 1, xs, ys


def line_ball(line, pitch):
    """(ball_x, ball_y) for one raw ATD line, in metres, NaN if the ball is not seen."""
    ball = line.split(FIELD_SEP)[2]
    if ball[0] == MISSING:
        return np.nan, np.nan
    return _xy(ball, *pitch)


def measured_player_speeds(px, py, fps):
    """Speed of every player slot per frame, m/s, over a SPEED_LAG window centred on the frame:
    frame t uses the positions at t - (SPEED_LAG - SPEED_LAG // 2) and t + SPEED_LAG // 2.

    NaN where the slot is untracked at either end, at the edges of the array, and above MAX_SPEED."""
    ahead = SPEED_LAG // 2
    speed = np.full(px.shape, np.nan)
    speed[SPEED_LAG - ahead:len(px) - ahead] = np.hypot(
        px[SPEED_LAG:] - px[:-SPEED_LAG], py[SPEED_LAG:] - py[:-SPEED_LAG]) * fps / SPEED_LAG
    with np.errstate(invalid="ignore"):
        speed[speed > MAX_SPEED] = np.nan
    return speed


def median_player_speed(px, py, fps):
    """Median speed of the measured players per frame, m/s, from (frames, slots) positions.

    NaN where fewer than MIN_PLAYERS are measured (see measured_player_speeds)."""
    speed = measured_player_speeds(px, py, fps)
    out = np.full(len(px), np.nan)
    enough = np.isfinite(speed).sum(axis=1) >= MIN_PLAYERS
    out[enough] = np.nanmedian(speed[enough], axis=1)
    return out


def speed_category(speed):
    """Category per frame: 0..N_SPEED_BINS-1 from standing to running, -1 where unmeasured."""
    cats = np.digitize(np.nan_to_num(speed), SPEED_EDGES).astype(np.int8)
    cats[np.isnan(speed)] = -1
    return cats


def read_distances(raw_path, team_of_track, n_frames, pitch, fps, keepers=(), halves=()):
    """Stream the raw ATD once. Frames with no tracking line count as ball missing.

    Returns (ball_x, ball_y, d_home, d_away, speed, stats): metres, NaN where not seen, and the
    median player speed in m/s. stats is (lines read, malformed lines skipped, frames with no
    line, keeper frames moved to the other team), reported so that a broken file does not pass
    for a long stretch of missing ball. A frame outside 0..n_frames-1 is an error, not something
    to index around. keepers are goalkeeper slots and halves (start, stop) frame slices; a keeper
    slot's team follows the end it stands in within each half (team_by_frame).
    """
    n_slots = len(team_of_track)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    has_line = np.zeros(n_frames, dtype=bool)
    n_lines = n_malformed = 0
    with open(raw_path) as handle:
        for line in handle:
            n_lines += 1
            if line.count(FIELD_SEP) != 2:
                n_malformed += 1
                continue
            frame, xs, ys = line_positions(line, n_slots, pitch)
            if not 0 <= frame < n_frames:
                raise ValueError("{}: frame_id {} is outside 1..{}".format(
                    raw_path, frame + 1, n_frames))
            px[frame], py[frame] = xs, ys
            ball_x[frame], ball_y[frame] = line_ball(line, pitch)
            has_line[frame] = True
    team = team_by_frame(px, team_of_track, keepers, halves, pitch[0])
    moved = int(np.sum(team != np.asarray(team_of_track, dtype=np.int8)))
    d_home, d_away = nearest_by_team(px, py, ball_x, ball_y, team)
    speed = median_player_speed(px.astype(float), py.astype(float), fps)
    return ball_x, ball_y, d_home, d_away, speed, (
        n_lines, n_malformed, int(np.sum(~has_line)), moved)


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


def cache_key(raw_path, team_of_track, n_frames, pitch, fps, keepers=(), halves=()):
    """Short hash of everything the distances depend on: parser version, the raw file (by size
    and modification time), the home/away slot map, the frame count, the pitch, the fps, the
    keeper slots and the halves."""
    stat = os.stat(raw_path)
    digest = hashlib.sha1(repr((CLUES_VERSION, os.path.basename(raw_path), stat.st_size,
                                int(stat.st_mtime), n_frames, tuple(pitch), fps,
                                tuple(keepers), tuple(map(tuple, halves)))).encode())
    digest.update(np.asarray(team_of_track, dtype=np.int8).tobytes())
    return digest.hexdigest()[:12]


def load_distances(raw_path, team_of_track, n_frames, pitch, fps, cache_dir, keepers=(),
                   halves=()):
    """read_distances, cached as .npz under cache_dir (data/, never committed).

    The file name carries cache_key(), so changing the parser, the data, the slot map, the pitch,
    the keepers or the halves writes a new cache instead of silently reading a stale one.
    """
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "hmm_distances_{}.npz".format(
            cache_key(raw_path, team_of_track, n_frames, pitch, fps, keepers, halves)))
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (cached["ball_x"], cached["ball_y"], cached["d_home"], cached["d_away"],
                    cached["speed"], tuple(int(x) for x in cached["stats"]))
    ball_x, ball_y, d_home, d_away, speed, stats = read_distances(
        raw_path, team_of_track, n_frames, pitch, fps, keepers, halves)
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, ball_x=ball_x, ball_y=ball_y, d_home=d_home,
                            d_away=d_away, speed=speed, stats=np.array(stats))
    return ball_x, ball_y, d_home, d_away, speed, stats
