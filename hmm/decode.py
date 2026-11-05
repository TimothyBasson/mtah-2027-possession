"""Sticky HMM v3: nearest-player gap + where the ball is + how fast the players move, decoded with
Viterbi, one path per half.

v2 added a ball that goes out of play: a missing ball on its own is close to neutral, and D comes
from the ball being seen past a line, or lost at or past one. v3 adds the players' speed. When the
ball is dead the players walk; in play most of them run. That catches stoppages with the ball in
view (fouls, free kicks), and says what a missing ball means: missing while the players walk leans
D, missing while they run leans live -- without assuming either for every dropout, which is where
v1 (missing = D) and v2 (missing = neutral) each went wrong on some of the external matches.

The gap clue counts a goalkeeper slot near a goal for the team defending that goal, not the team
the metadata gives the slot: the tracker sometimes carries one keeper under the other keeper's
slot (hmm/clues.py).

    python -m hmm.decode                    # writes evaluation/results/hmm_v3.csv
    python -m hmm.decode --baum-welch       # also prints what unsupervised EM would change
    python -m hmm.decode --baum-welch --bw-weight 0.1   # ... with the decoder's tempered emissions

No number below is tuned on the public window -- EVALUATION.md shows that tuning there picks the
wrong setting. Most are set by hand; where one was checked against this match, the data was
label-free: summaries of its own tracking (how often the ball is missing, where it was last seen)
and Smart Tagging, which is Metrica's automatic output, not ground truth. The speed table is the
one exception to "by hand": it is counted from the labelled external matches (external/, other
games, which RULES.md allows). Each comment below says which. Baum-Welch on the full match (no labels) is run only as a check on these
numbers, and does not overwrite them.

numpy only, like starter/baseline.py, so it runs in the scorer's environment.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from hmm.clues import (  # noqa: E402
    KEEPER_ON_BALL, N_GAP_BINS, ball_is_usable, ball_place_category, gap_category, keeper_slots,
    load_distances, speed_category)
from hmm.model import (  # noqa: E402
    A, D, H, STATES, baum_welch, build_log_emit, rate_log_trans, viterbi)
from starter.baseline import RESULTS_DIR, team_of_each_track, write_submission  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]

# p(gap bin | H), bins from "away much closer" to "home much closer" (see hmm/clues.py), on
# frames with a usable ball. The team in possession is usually, not always, nearest -- a pass in
# flight or a tackle puts the other side closer. A mirrors H. During a stoppage the gap says
# nothing about the state: flat.
GAP_GIVEN_HOME = np.array([0.05, 0.10, 0.20, 0.30, 0.35])
GAP_GIVEN_DEAD = np.full(N_GAP_BINS, 1.0 / N_GAP_BINS)

# p(ball place | state), columns IN_PLAY, OUTSIDE, MISSING_AFTER_LINE, MISSING_AFTER_PLAY,
# MISSING_STALE (hmm/clues.py). The ball is missing on 41% of graded frames, in live play and
# stoppages alike, so missing in itself is near neutral (40% live, 55% dead). What separates them:
# - OUTSIDE: in play the tracked ball crosses a line only by noise or for an instant; when it has
#   gone out it is often seen past the line, before the tracker drops it or while it is fetched.
# - MISSING_AFTER_LINE: in play the ball is within 2 m of a line on ~9% of frames, so few dropouts
#   start there; a ball that goes out is usually lost right at the line.
# - MISSING_AFTER_PLAY leans slightly live.
# - MISSING_STALE says nothing on purpose: a long dropout is as likely to be live play as a
#   stoppage (of the 44 dropouts over 20 s, about a third start at a line), and leaning it towards
#   D would bring back v1's failure.
# The shape is football reasoning; the levels were checked against the full match's own
# label-free summaries (41% missing, 9% of sightings within 2 m of a line, 25-43% of dropouts
# starting there), and report_shares prints the observed shares on every run.
#
# Not double counting with the gap clue: the gap table is p(gap | state, ball usable) and is only
# observed when the ball is usable, so the two tables factor p(place, gap | state) by the chain
# rule. IN_PLAY's 0.59 vs 0.41 is p(ball usable | state), which the gap table does not contain.
PLACE_GIVEN_LIVE = np.array([0.590, 0.003, 0.040, 0.217, 0.150])
PLACE_GIVEN_DEAD = np.array([0.410, 0.040, 0.250, 0.150, 0.150])

# p(median player speed bin | state), bins (hmm/clues.py SPEED_EDGES) <0.6, <0.9, <1.2, <1.5,
# <2.0, <2.5, >=2.5 m/s. Counted on the labelled external matches: the mean over the 7 IDSSE
# matches and the mean over the 3 Metrica samples, averaged so each provider counts once. The two
# providers agree to within 0.03 in most bins (IDSSE live has fewer standing frames, 0.002 against
# 0.036). H and A share the live row: speed says the ball is in play, not who has it.
SPEED_GIVEN_LIVE = np.array([0.020, 0.035, 0.100, 0.167, 0.213, 0.148, 0.317])
SPEED_GIVEN_DEAD = np.array([0.222, 0.221, 0.247, 0.208, 0.076, 0.013, 0.013])

# How long a last sighting at the line is carried into a missing stretch; past this the decoder
# relies on the transitions alone. Football reasoning: a throw-in is taken within ~10 s, a goal
# kick or corner usually within 20-40 s of the ball going out. Checked against Smart Tagging (not
# ground truth): at the tagged corners the ball was last seen a median 24 s before the kick, so
# 15 s ended the stoppage early; 30 s catches more tagged corners and goal kicks, and 45 s, 60 s or
# the whole stretch catch no more -- they only stretch the longest stoppages.
CARRY_SECONDS = 30

# Trust exponent on each frame's clue. At 25 fps neighbouring frames repeat the same look at the
# ball; 0.1 counts ~10 frames (0.4 s) as one independent observation.
WEIGHT = 0.1

# Mean possession and stoppage length, and the share of possessions that end in a stoppage
# rather than going straight to the other team.
POSSESSION_SECONDS = 12.0
STOPPAGE_SECONDS = 8.0
TO_DEAD_SHARE = 0.4

# Each graded half opens just before a kickoff, i.e. out of a stoppage.
START = np.array([0.1, 0.1, 0.8])

# The ball is not in play before a kickoff, so graded frames before one are D. The second half's
# graded block opens ~2 minutes before its kickoff, and a soft start prior alone lets the decoder
# fill that stretch with H.
#
# The kickoff frames are the published ones (DATA.md, FAQ.md, starter/load_smart_tagging.py):
# 11185, the first graded frame, and 113742, so 110808..113741 -- the 2,934 frames FAQ.md says are
# all D -- are pinned and the kickoff frame itself is live. Not the games.py playback anchors
# (11186, 113741), which are one frame off in opposite directions.
KICKOFFS = tuple(KICKOFF_VIDEO_FRAMES[half] for half in ("1st Half", "2nd Half"))

# A log-likelihood low enough to rule a state out without the inf - inf of a true -inf.
IMPOSSIBLE = -1e9

CACHE_DIR = os.path.join(DATA_DIR, "cache")


def gap_table():
    """log p(gap bin | state), states x gap bins."""
    return np.log(np.array([GAP_GIVEN_HOME, GAP_GIVEN_HOME[::-1], GAP_GIVEN_DEAD]))


def place_table():
    """log p(ball place | state), states x ball-place categories."""
    return np.log(np.array([PLACE_GIVEN_LIVE, PLACE_GIVEN_LIVE, PLACE_GIVEN_DEAD]))


def speed_table(live=SPEED_GIVEN_LIVE, dead=SPEED_GIVEN_DEAD):
    """log p(speed bin | state), states x speed bins."""
    return np.log(np.array([live, live, dead]))


def transitions():
    return rate_log_trans(POSSESSION_SECONDS * FPS, STOPPAGE_SECONDS * FPS, TO_DEAD_SHARE)


def graded_segments(game=GAME):
    """(start, stop) video-frame slices of the graded halves, stop exclusive."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def pin_dead_before_kickoff(log_emit, lo, hi, kickoffs):
    """Rule out H and A on the frames of [lo, hi) that come before a kickoff inside it."""
    for kickoff in kickoffs:
        if lo < kickoff < hi:
            log_emit[:kickoff - lo, [H, A]] = IMPOSSIBLE
    return log_emit


def decode(clues, log_trans, log_start, segments, kickoffs=KICKOFFS):
    """Labels for every video frame: one Viterbi path per half, X everywhere else.

    clues is a list of (obs, log_table), each obs covering every video frame."""
    labels = np.full(len(clues[0][0]), "X", dtype="<U1")
    for lo, hi in segments:
        log_emit = build_log_emit([(obs[lo:hi], table, WEIGHT) for obs, table in clues])
        pin_dead_before_kickoff(log_emit, lo, hi, kickoffs)
        labels[lo:hi] = STATES[viterbi(log_emit, log_trans, log_start)]
    return labels


def load_observations(game=GAME, data_dir=None):
    """(gap, place, speed) categories for every video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    print("reading {} (cached after the first run) ...".format(os.path.basename(raw_path)))
    pitch = GAMES[game]["pitch"]
    keepers = keeper_slots(meta_path)
    ball_x, ball_y, d_home, d_away, speed, stats = load_distances(
        raw_path, team_of_each_track(meta_path), n_frames, pitch, FPS, CACHE_DIR,
        keepers=keepers, halves=graded_segments(game))
    print("  {} tracking lines, {} malformed and skipped, {} frames with no line "
          "(counted as ball missing)".format(*stats[:3]))
    print("  keeper slots {}: {} frames counted for the team defending the goal they stand by, "
          "{} of them within {:.0f} m of the ball".format(
              [slot + 1 for slot in keepers], stats[3], stats[4], KEEPER_ON_BALL))
    if keepers and stats[3] == 0:
        # team_by_frame fell back to the metadata's map (its warning is not repeated when the
        # distances come from the cache), so the keeper fix is off for this run.
        print("WARNING: the keeper rule moved no frames -- it fell back to the metadata's teams. "
              "Check the keeper slots and that the teams switch ends.", file=sys.stderr)
    gap = gap_category(ball_is_usable(ball_x, ball_y, pitch), d_home, d_away)
    place = ball_place_category(ball_x, ball_y, pitch, CARRY_SECONDS * FPS)
    return gap, place, speed_category(speed)


PLACE_NAMES = ["in play", "outside", "lost@line", "lost@play", "stale"]
SPEED_NAMES = ["<0.6", "<0.9", "<1.2", "<1.5", "<2.0", "<2.5", ">=2.5"]


def report_shares(obs, names, what, segments):
    """Share of each category over the graded frames: label-free, a check on the PLACE_GIVEN_* and
    SPEED_GIVEN_* tables (mixed at the decoded D share they should land near these)."""
    graded = np.concatenate([obs[lo:hi] for lo, hi in segments])
    print("{} on graded frames: ".format(what) + "  ".join(
        "{} {:.1%}".format(name, np.mean(graded == k)) for k, name in enumerate(names)))


def summarise(labels, segments):
    """Label counts and possession counts per half: what the file claims, no ground truth."""
    for half, (lo, hi) in enumerate(segments, 1):
        seg = labels[lo:hi]
        runs = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], runs))
        kinds = seg[starts]
        lengths = np.diff(np.concatenate((starts, [len(seg)])))
        possessions = lengths[kinds != "D"]
        stoppages = lengths[kinds == "D"]
        print("half {}: H {:>6}  A {:>6}  D {:>6} (D {:.0%})".format(
            half, *(int(np.sum(seg == s)) for s in "HAD"), np.mean(seg == "D")))
        for name, spells in (("possessions", possessions), ("stoppages", stoppages)):
            if len(spells):
                print("  {:>4} {:<11} median {:5.1f} s  mean {:5.1f} s  p90 {:5.1f} s  "
                      "max {:5.0f} s".format(len(spells), name, *(np.array(
                          [np.median(spells), spells.mean(), np.percentile(spells, 90),
                           spells.max()]) / FPS)))


def report_baum_welch(clues, log_trans, segments, n_iter, weight=1.0):
    """Fit EM from the hand-set numbers and print where it disagrees. Changes nothing.

    With weight < 1 the E-step uses the same tempered emissions as the decoder, but the M-step
    re-estimates the table as if they were untempered, so it is a heuristic rather than true EM.
    It shows where the decoder's own posteriors pull the numbers, not a maximum-likelihood fit.
    """
    fit_tables, fit_trans, history = baum_welch(clues, log_trans, np.log(START), n_iter=n_iter,
                                                weight=weight, segments=segments)
    print("\nBaum-Welch, {} iterations, emission weight {}, log-likelihood {:.0f} -> {:.0f}".format(
        n_iter, weight, history[0], history[-1]))
    gap_names = ["gap{}".format(i) for i in range(N_GAP_BINS)]
    for names, (_, log_table), fit_table in zip((gap_names, PLACE_NAMES, SPEED_NAMES), clues,
                                                fit_tables):
        print("emissions       " + " ".join("{:>9}".format(n) for n in names))
        for state in (H, A, D):
            for tag, table in (("hand", log_table), ("fit ", fit_table)):
                print("  {} {}     ".format(STATES[state], tag)
                      + " ".join("{:9.3f}".format(p) for p in np.exp(table[state])))
    hand, fit = np.exp(log_trans), np.exp(fit_trans)
    for state in (H, A, D):
        line = "  {} mean run: hand {:6.1f} s, fit {:6.1f} s".format(
            STATES[state], 1 / (1 - hand[state, state]) / FPS, 1 / (1 - fit[state, state]) / FPS)
        if state != D:
            line += ";  share of exits to D: hand {:.2f}, fit {:.2f}".format(
                hand[state, D] / (1 - hand[state, state]), fit[state, D] / (1 - fit[state, state]))
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "hmm_v3.csv"))
    parser.add_argument("--baum-welch", type=int, nargs="?", const=20, default=0,
                        metavar="N_ITER", help="also run unsupervised EM as a check")
    parser.add_argument("--bw-weight", type=float, default=1.0, metavar="W",
                        help="emission weight for the EM check: 1.0 is plain EM, {} matches "
                             "the decoder (tempered, a heuristic)".format(WEIGHT))
    args = parser.parse_args()

    gap, place, speed = load_observations()
    segments = graded_segments()
    report_shares(place, PLACE_NAMES, "ball place", segments)
    report_shares(speed, SPEED_NAMES, "player speed (m/s)", segments)
    clues = [(gap, gap_table()), (place, place_table()), (speed, speed_table())]
    log_trans = transitions()

    labels = decode(clues, log_trans, np.log(START), segments)
    path = write_submission(args.out, labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    summarise(labels, segments)

    if args.baum_welch:
        report_baum_welch(clues, log_trans, segments, args.baum_welch, args.bw_weight)


if __name__ == "__main__":
    main()
