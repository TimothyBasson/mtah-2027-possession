"""Sticky HMM v5: nearest-player gap and distance + where the ball is + how fast the players move +
who kicked the ball in flight, decoded with Viterbi, one path per half.

v2 added a ball that goes out of play: a missing ball on its own is close to neutral, and D comes
from the ball being seen past a line, or lost at or past one. v3 adds the players' speed. When the
ball is dead the players walk; in play most of them run. That catches stoppages with the ball in
view (fouls, free kicks), and says what a missing ball means: missing while the players walk leans
D, missing while they run leans live -- without assuming either for every dropout, which is where
v1 (missing = D) and v2 (missing = neutral) each went wrong on some of the external matches.

v4 adds the flight clue (hmm/ball.py). A long ball or clearance flies over the other team, so the
nearest player during the flight is often an opponent, and the gap clue switched possession at the
kick instead of at the landing. Now every touch the ball's velocity shows starts a flight, the
toucher's team is the clue until the next touch. On the external matches this lifts spellF1 on all
10 (external/score_hmm.py).

v5 changes three things, each measured on the external matches (external/score_hmm.py), where
together they lift spellF1 on all 10:
- The trust in each frame's clues doubles (WEIGHT 0.1 -> 0.2). v4 found about 200 possessions in
  every match, whatever the labels said (315-511), and almost none of those under 4 s.
- The gap is read together with the distance from the ball to the nearest player
  (hmm/clues.py gap_distance_category), with a table counted from the external labels like the
  speed and flight tables. A gap with a player on the ball says more than the same gap with
  nobody within 5 m, and nobody near the ball is itself a sign of a stoppage.
- The ball-place table is no longer set by hand. What a missing ball means depends on the
  tracking: this provider loses the ball mostly in stoppages, while a ball knocked out at random
  says nothing. So the table is counted on the match being decoded, from what the other three
  clues say about each frame (fit_place) -- no labels, and the same code on every match.

The gap clue counts a goalkeeper slot near a goal for the team defending that goal, not the team
the metadata gives the slot: the tracker sometimes carries one keeper under the other keeper's
slot (hmm/clues.py).

    python -m hmm.decode                    # writes evaluation/results/hmm_v5.csv
    python -m hmm.decode --baum-welch       # also prints what unsupervised EM would change
    python -m hmm.decode --baum-welch --bw-weight 0.2   # ... with the decoder's tempered emissions

No number below is tuned on the public window -- EVALUATION.md shows that tuning there picks the
wrong setting. Most are set by hand; where one was checked against this match, the data was
label-free: summaries of its own tracking (how often the ball is missing, where it was last seen)
and Smart Tagging, which is Metrica's automatic output, not ground truth. The gap-distance, speed
and flight tables are the exceptions to "by hand": they are counted from the labelled external
matches (external/, other games, which RULES.md allows), and WEIGHT was chosen there. The place
table is counted on this match without labels (fit_place), which RULES.md allows as learning
without human labels. Each comment below says which. Baum-Welch on the full match (no labels) is
run only as a check on these numbers, and does not overwrite them.

numpy only, like starter/baseline.py, so it runs in the scorer's environment.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from hmm.ball import flight_clue, mirror_flight  # noqa: E402
from hmm.clues import (  # noqa: E402
    KEEPER_ON_BALL, N_DISTANCE_BINS, N_GAP_BINS, N_PLACES, ball_is_usable, ball_place_category,
    gap_distance_category, keeper_slots, load_distances, mirror_gap_distance, speed_category)
from hmm.model import (  # noqa: E402
    A, D, H, STATES, baum_welch, build_log_emit, forward_backward, rate_log_trans, viterbi)
from starter.baseline import RESULTS_DIR, team_of_each_track, write_submission  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]

# p(gap bin | H), bins from "away much closer" to "home much closer" (see hmm/clues.py), on
# frames with a usable ball. The team in possession is usually, not always, nearest -- a pass in
# flight or a tackle puts the other side closer. A mirrors H. During a stoppage the gap says
# nothing about the state: flat. Set by hand for v1-v4, which external/score_hmm.py still scores;
# v5 reads the gap-distance table below instead. (Counted from the external labels the H row comes
# out close to this one, and using it moved spellF1 by 0.006 at most.)
GAP_GIVEN_HOME = np.array([0.05, 0.10, 0.20, 0.30, 0.35])
GAP_GIVEN_DEAD = np.full(N_GAP_BINS, 1.0 / N_GAP_BINS)

# p(gap bin, distance bin | state), on frames with a usable ball. Rows are the gap bins, from "away
# much closer" to "home much closer"; columns the distance from the ball to the nearest player of
# either team, <1, <2.5, <5, >=5 m (hmm/clues.py DISTANCE_EDGES). Counted on the labelled external
# matches like the speed table: the mean over the 7 IDSSE matches (ball knocked out to 59%) and the
# mean over the 3 Metrica samples, averaged so each provider counts once
# (python external/score_hmm.py --tables prints them). A mirrors H, and the D rows are counted
# with the teams swapped as well, so D never favours a team.
# What it holds that the hand-set gap table did not: with a home player within 1 m of the ball
# and away more than 3 m further off, H is 37 times likelier than A (the hand-set table: 7); with
# the nearest player 5 m or more away the same gap makes it 2.2; and with nobody within 5 m, D is
# likelier than either team. The providers differ in the home-much-closer row: on the Metrica
# samples, whose ball sits on the players, the nearest player is within 1 m on 0.36 of H frames
# and 1-2.5 m away on 0.06; on IDSSE 0.23 and 0.12. Elsewhere they agree to within 0.02.
GAP_DISTANCE_GIVEN_HOME = np.array([
    [0.008, 0.016, 0.022, 0.013],
    [0.009, 0.022, 0.032, 0.020],
    [0.049, 0.050, 0.054, 0.033],
    [0.107, 0.045, 0.038, 0.022],
    [0.294, 0.088, 0.050, 0.029],
])
GAP_DISTANCE_GIVEN_DEAD = np.array([
    [0.023, 0.050, 0.072, 0.120],
    [0.019, 0.020, 0.037, 0.053],
    [0.045, 0.030, 0.064, 0.072],
    [0.019, 0.020, 0.037, 0.053],
    [0.023, 0.050, 0.072, 0.120],
])

# p(ball place | state) as v2-v4 had it, set by hand; external/score_hmm.py still scores them with
# it. v5 counts the table on the match itself (fit_place, below), because the central assumption
# here turned out to depend on the tracking: "the ball is missing in live play and stoppages
# alike" holds for a ball knocked out at random, as on the IDSSE matches, but not for this
# provider. On its three sample matches the ball is seen on 88% of live frames and 4% of dead
# ones, and on BAR-ATH the ball goes missing in runs of a median 10.7 s, 1.4 a minute, with the
# players walking on 64% of those frames against 20% of the frames with the ball in view.
#
# Columns IN_PLAY, OUTSIDE, MISSING_AFTER_LINE, MISSING_AFTER_PLAY, MISSING_STALE (hmm/clues.py).
# The ball is missing on 41% of graded frames, and v2-v4 took missing in itself as near neutral
# (40% live, 55% dead). What separates them:
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

# p(flight category | state, in flight), categories (hmm/ball.py) kicker home in flight phases
# <0.5 s, 0.5-1.5 s, >=1.5 s since the kick, the same for kicker away, then kicker unknown.
# Counted on the labelled external matches like the speed table (IDSSE with the ball knocked out
# to 59%, as the model sees it), so each provider counts once. A mirrors H. D is the mean of the
# two (flight_table): a flight in a stoppage is rare, and the few such frames are no basis for a
# row of their own.
# Counted with the 3 s flight cap (hmm/ball.py FLIGHT_SECONDS). The providers agree to within 0.08
# in every bin; the unknown-kicker share is larger on IDSSE (0.07 against 0.01) because its ball is
# knocked out. The kicker's team is 23-28 times likelier than the other team in every phase, so the
# later phases are not weaker in the labels.
FLIGHT_GIVEN_HOME = np.array([0.365, 0.412, 0.146, 0.016, 0.015, 0.006, 0.040])

# How long a last sighting at the line is carried into a missing stretch; past this the decoder
# relies on the transitions alone. Football reasoning: a throw-in is taken within ~10 s, a goal
# kick or corner usually within 20-40 s of the ball going out. Checked against Smart Tagging (not
# ground truth): at the tagged corners the ball was last seen a median 24 s before the kick, so
# 15 s ended the stoppage early; 30 s catches more tagged corners and goal kicks, and 45 s, 60 s or
# the whole stretch catch no more -- they only stretch the longest stoppages.
CARRY_SECONDS = 30

# The lowest share fit_place leaves any ball-place category in either row. Without it a category
# the other clues never saw in live play (a stale ball, on some matches) would rule live play out
# on its own; with it no single frame's place is worth more than a factor of 1 / PLACE_FLOOR.
PLACE_FLOOR = 0.02

# Trust exponent on each frame's clue. At 25 fps neighbouring frames repeat the same look at the
# ball; 0.2 counts 5 frames (0.2 s) as one independent observation. v1-v4 used 0.1. Chosen on the
# 10 external matches, where spellF1 with the v4 clues is 0.443 at 0.1, 0.477 at 0.15, 0.486 at
# 0.2, 0.489 at 0.3 and 0.474 at 0.5, with both providers peaking between 0.2 and 0.3. 0.2 is the
# cautious end of that: BAR-ATH's flight clue is noisier than the external one (hmm/ball.py), and
# a higher weight trusts it more.
WEIGHT = 0.2
# What v2-v4 were built and reported with; external/score_hmm.py scores them at it.
WEIGHT_V4 = 0.1

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


def gap_distance_table(home=GAP_DISTANCE_GIVEN_HOME, dead=GAP_DISTANCE_GIVEN_DEAD):
    """log p(gap bin, distance bin | state), states x gap-distance categories. A mirrors H."""
    home = np.ravel(home) / np.sum(home)  # the rounded constants do not sum to exactly 1
    dead = np.ravel(dead) / np.sum(dead)
    return np.log(np.array([home, mirror_gap_distance(home), dead]))


def place_table(live=PLACE_GIVEN_LIVE, dead=PLACE_GIVEN_DEAD):
    """log p(ball place | state), states x ball-place categories. H and A share the live row."""
    return np.log(np.array([live, live, dead]))


def speed_table(live=SPEED_GIVEN_LIVE, dead=SPEED_GIVEN_DEAD):
    """log p(speed bin | state), states x speed bins."""
    return np.log(np.array([live, live, dead]))


def flight_table(home=FLIGHT_GIVEN_HOME):
    """log p(flight category | state, in flight), states x categories. A mirrors H, and D is
    their average: in flight, the clue tells the teams apart, not live from dead."""
    home = np.asarray(home) / np.sum(home)  # the rounded constants sum to 0.999
    away = mirror_flight(home)
    return np.log(np.array([home, away, (home + away) / 2]))


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


def fit_place(place, other_clues, log_trans, log_start, segments, kickoffs=KICKOFFS,
              weight=WEIGHT):
    """(live, dead) p(ball place | state) for one match, counted on the match itself, no labels.

    The other clues -- gap and distance, player speed, flights -- give each frame a probability of
    being live or dead (forward-backward, with the decoder's transitions and weight). The place
    categories are then counted with those probabilities as weights. The place clue takes no part
    in its own labelling and the count is made once, not iterated, so it cannot talk itself into
    "dead means the ball is missing": it only records how the ball's place lines up with what the
    players and the ball's motion already say. Each share is at least PLACE_FLOOR.

    Checked on the external matches, where it is fitted on each match in the same way: on IDSSE,
    where the ball is knocked out at random, it comes out close to the old hand-set table (ball
    seen on 60% of live frames and 49% of dead ones) and the scores do not move; on the Metrica
    samples it finds the ball seen on 91% of live frames and 15% of dead ones, and spellF1 rises
    (0.742 -> 0.774). On BAR-ATH it finds 83% and 25%; main() prints both rows on every run.
    """
    counts = np.zeros((2, N_PLACES))
    for lo, hi in segments:
        log_emit = build_log_emit([(obs[lo:hi], table, weight) for obs, table in other_clues])
        pin_dead_before_kickoff(log_emit, lo, hi, kickoffs)
        post, _, _ = forward_backward(log_emit, log_trans, log_start)
        np.add.at(counts[0], place[lo:hi], post[:, H] + post[:, A])
        np.add.at(counts[1], place[lo:hi], post[:, D])
    shares = np.maximum(counts / counts.sum(axis=1, keepdims=True), PLACE_FLOOR)
    shares /= shares.sum(axis=1, keepdims=True)
    return shares[0], shares[1]


def decode(clues, log_trans, log_start, segments, kickoffs=KICKOFFS, weight=WEIGHT):
    """Labels for every video frame: one Viterbi path per half, X everywhere else.

    clues is a list of (obs, log_table), each obs covering every video frame."""
    labels = np.full(len(clues[0][0]), "X", dtype="<U1")
    for lo, hi in segments:
        log_emit = build_log_emit([(obs[lo:hi], table, weight) for obs, table in clues])
        pin_dead_before_kickoff(log_emit, lo, hi, kickoffs)
        labels[lo:hi] = STATES[viterbi(log_emit, log_trans, log_start)]
    return labels


def load_observations(game=GAME, data_dir=None):
    """(gap-distance, place, speed, flight) categories for every video frame."""
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
    usable = ball_is_usable(ball_x, ball_y, pitch)
    gap = gap_distance_category(usable, d_home, d_away)
    place = ball_place_category(ball_x, ball_y, pitch, CARRY_SECONDS * FPS)
    # the flights use the same usable ball as the gap clue: a spare ball far outside is no flight
    flight, touches = flight_clue(np.where(usable, ball_x, np.nan),
                                  np.where(usable, ball_y, np.nan), d_home, d_away, FPS)
    print("  {} touches, {:.0%} with a player within reach; {:.1%} of frames in flight".format(
        len(touches["frame"]), np.mean(touches["team"] >= 0), np.mean(flight >= 0)))
    return gap, place, speed_category(speed), flight


PLACE_NAMES = ["in play", "outside", "lost@line", "lost@play", "stale"]
SPEED_NAMES = ["<0.6", "<0.9", "<1.2", "<1.5", "<2.0", "<2.5", ">=2.5"]
FLIGHT_NAMES = ["H<0.5s", "H<1.5s", "H>=1.5s", "A<0.5s", "A<1.5s", "A>=1.5s", "unknown"]


def report_shares(obs, names, what, segments):
    """Share of each category over the graded frames: label-free, a check on the fitted place
    table and on SPEED_GIVEN_* (mixed at the decoded D share they should land near these)."""
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
    gap_names = ["g{}d{}".format(g, d) for g in range(N_GAP_BINS) for d in range(N_DISTANCE_BINS)]
    for names, (_, log_table), fit_table in zip((gap_names, PLACE_NAMES, SPEED_NAMES,
                                                 FLIGHT_NAMES), clues,
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
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "hmm_v5.csv"))
    parser.add_argument("--baum-welch", type=int, nargs="?", const=20, default=0,
                        metavar="N_ITER", help="also run unsupervised EM as a check")
    parser.add_argument("--bw-weight", type=float, default=1.0, metavar="W",
                        help="emission weight for the EM check: 1.0 is plain EM, {} matches "
                             "the decoder (tempered, a heuristic)".format(WEIGHT))
    args = parser.parse_args()

    gap, place, speed, flight = load_observations()
    segments = graded_segments()
    report_shares(place, PLACE_NAMES, "ball place", segments)
    report_shares(speed, SPEED_NAMES, "player speed (m/s)", segments)
    report_shares(flight, FLIGHT_NAMES, "flight", segments)
    log_trans = transitions()
    others = [(gap, gap_distance_table()), (speed, speed_table()), (flight, flight_table())]
    live, dead = fit_place(place, others, log_trans, np.log(START), segments)
    for name, row in (("live", live), ("dead", dead)):
        print("ball place given {}, fitted: ".format(name) + "  ".join(
            "{} {:.1%}".format(n, p) for n, p in zip(PLACE_NAMES, row)))
    # gap, place, speed, flight: the order report_baum_welch names them in
    clues = [others[0], (place, place_table(live, dead))] + others[1:]

    labels = decode(clues, log_trans, np.log(START), segments)
    path = write_submission(args.out, labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    summarise(labels, segments)

    if args.baum_welch:
        report_baum_welch(clues, log_trans, segments, args.baum_welch, args.bw_weight)


if __name__ == "__main__":
    main()
