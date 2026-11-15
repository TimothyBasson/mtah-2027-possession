"""Sticky HMM v6: nearest-player gap and distance + where the ball is + how fast the players move +
who kicked the ball in flight + which team is spread wider, decoded with Viterbi, one path per half.

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

v5 changes two things, each measured on the external matches (external/score_hmm.py), where each
helps on both providers and together they lift spellF1 on all 10:
- The trust in each frame's clues doubles (WEIGHT 0.1 -> 0.2). v4 found about 200 possessions in
  every match, whatever the labels said (315-511), and almost none of those under 4 s.
- The gap is read together with the distance from the ball to the nearest player
  (hmm/clues.py gap_distance_category), with a table counted from the external labels like the
  speed and flight tables. A gap with a player on the ball says more than the same gap with
  nobody within 5 m, and nobody near the ball is itself a sign of a stoppage.
A third change was built and is kept behind --fit-place, not in the default: counting the
ball-place table on the match being decoded, from what the other three clues say about each frame
(fit_place), instead of setting it by hand. What a missing ball means depends on the tracking, and
on this match the fit finds the ball seen on 85% of live frames and 27% of dead ones. But against
the hand-set table it is +0.007 spellF1 over the 10 external matches with an interval that spans
zero (up on the 3 Metrica samples, slightly down on 6 of the 7 IDSSE matches), and nothing we may
use on BAR-ATH confirms its dead row. With no evidence to separate them, the default is the one
with fewer assumptions. fit_place's docstring has the numbers.

v6 adds the width clue (hmm/clues.py team_width_difference): the team with the ball spreads out
across the pitch and the team without it closes up. It reads the players only, so it speaks on the
frames where the ball is missing, which is where v5 names the wrong team most often. It changes
slowly, so each frame adds little that the last did not, and it gets a quarter of the other clues'
weight (WIDTH_WEIGHT). On the external matches it lifts spellF1 on all three sets, by 0.01-0.02
(external/README.md).

v7 adds the control clue (hmm/control.py): a gradient-boosted classifier, trained on the 30
external matches (external/train_control.py), reads both teams' nearest players, how they move with
the ball, the touches either side and windows before and after each frame, and says how likely
H, A and D are. Its log posterior over the class shares is added to every v6 clue at
CONTROL_WEIGHT. With the classifier trained with each set left out, it lifts spellF1 on all three
sets against both the raw and the control-folded labels (external/score_control.py). Two things
tried and not kept: letting it replace the gap-distance and flight clues (worse everywhere), and
making a change of team cheaper where a second classifier sees one (helps the raw labels, not the
folded ones, and costs SkillCorner). The trees run in numpy (hmm/trees.py); only training needs
scikit-learn.

v8 changes what the control classifier reads and how it is trained (hmm/control.py
joint_features, external/train_control.py):
- It no longer reads what the ball does on its own (seen or not, its speed). Two of the three
  outside providers freeze or hide the ball in a stoppage, and the classifier had learned that.
- It also reads the team-shape features of seq/features.py: all 22 players, not only the ones by
  the ball, so it has something to go on where the ball is missing or noisy.
- The three providers count equally in training, the trees are regularised, and nothing is
  stopped early.
- Optional and off by default: a decoded possession under --min-possession seconds with the
  other team on both sides goes back to that team (fold_brief_possessions).
With the classifier trained with each set left out, spellF1 against the control-folded labels goes
from v7's 0.609 / 0.798 / 0.428 to 0.644 / 0.816 / 0.480 (external/README.md).

The gap clue counts a goalkeeper slot near a goal for the team defending that goal, not the team
the metadata gives the slot: the tracker sometimes carries one keeper under the other keeper's
slot (hmm/clues.py).

    python -m hmm.decode                    # writes evaluation/results/hmm_v8.csv
    python -m hmm.decode --no-control       # v6 again (v7 is commit fe457a3: its classifier read
                                            # columns the v8 features leave out)
    python -m hmm.decode --no-control --no-width   # v5 again
    python -m hmm.decode --weight 0.1 --hand-gap   # v4 again; either flag alone takes that one
                                            # change back out, to tell them apart
    python -m hmm.decode --fit-place        # with the place table counted on the match
    python -m hmm.decode --placebo          # also checks fit_place against a shifted ball
    python -m hmm.decode --baum-welch       # also prints what unsupervised EM would change
    python -m hmm.decode --baum-welch --bw-weight 0.2   # ... with the decoder's tempered emissions

No number below is tuned on the public window -- EVALUATION.md shows that tuning there picks the
wrong setting. Most are set by hand; where one was checked against this match, the data was
label-free: summaries of its own tracking (how often the ball is missing, where it was last seen)
and Smart Tagging, which is Metrica's automatic output, not ground truth. The gap-distance, speed
and flight tables are the exceptions to "by hand": they are counted from the labelled external
matches (external/, other games, which RULES.md allows), and WEIGHT was chosen there. (With
--fit-place the place table is counted on this match without labels, which RULES.md allows as
learning without human labels.) Each comment below says which. Baum-Welch on the full match (no
labels) is run only as a check on these numbers, and does not overwrite them.

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
from hmm import trees  # noqa: E402
from hmm.clues import (  # noqa: E402
    KEEPER_ON_BALL, MISSING_AFTER_LINE, N_DISTANCE_BINS, N_GAP_BINS, N_PLACES, ball_is_usable,
    ball_place_category, cache_key, gap_distance_category, keeper_slots, load_distances,
    mirror_gap_distance, read_positions, speed_category, team_by_frame, width_category)
from hmm.control import control_log_ratio, hold, joint_features  # noqa: E402
from hmm.model import (  # noqa: E402
    A, D, H, STATES, baum_welch, build_log_emit, forward_backward, rate_log_trans, viterbi)
from starter.baseline import RESULTS_DIR, team_of_each_track, write_submission  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

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

# p(ball place | state), set by hand, as v2-v5 have it.
#
# Its central assumption is the doubtful part: "the ball is missing in live play and stoppages
# alike" holds for a ball knocked out at random, as on the IDSSE matches, but probably not for
# this provider. On its three sample matches the ball is seen on 90% of live frames and 3% of dead
# ones, and on BAR-ATH the ball goes missing in runs of a median 10.7 s, 1.4 a minute, with the
# players walking on 64% of those frames against 20% of the frames with the ball in view. The
# speed clue already carries much of that (missing while the players walk leans D). fit_place,
# below, counts this table on the match instead; it is not the default because the external
# matches cannot tell the two apart and nothing we may use on BAR-ATH confirms the fitted one.
#
# Columns IN_PLAY, OUTSIDE, MISSING_AFTER_LINE, MISSING_AFTER_PLAY, MISSING_STALE (hmm/clues.py).
# The ball is missing on 41% of graded frames, and this table takes missing in itself as near
# neutral (40% live, 55% dead). What separates them:
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
# Not tuned: on the external matches spellF1 over all 10 is 0.521-0.522 for any floor from 0.001
# to 0.1.
PLACE_FLOOR = 0.02

# p(width category | H), away-much-wider .. home-much-wider (hmm/clues.py WIDTH_EDGES), and given D.
# Counted on the 30 external matches' labels, each of the three sets counting once
# (python external/score_hmm.py --tables prints them); no set's home row is further than 0.024 from
# this one. First counted with a floor of 3 tracked players; recounted at MIN_WIDTH_PLAYERS = 6,
# --tables prints the same rows to three decimals (within 0.001), so the constants stand. A mirrors H, and D is counted both ways round so it cannot favour a team.
WIDTH_GIVEN_HOME = np.array([0.015, 0.025, 0.051, 0.047, 0.061, 0.083, 0.106, 0.194, 0.217, 0.202])
WIDTH_GIVEN_DEAD = np.array([0.087, 0.100, 0.135, 0.092, 0.085, 0.085, 0.092, 0.135, 0.100, 0.087])
# Trust exponent on the width clue. A team's shape changes over seconds, not frames, so it gets
# less than the ball clues' WEIGHT. This number is tuned on the external matches it is judged on:
# spellF1 gain over v5 at 0.02 / 0.035 / 0.05 / 0.075 / 0.1 / 0.15 / 0.2 is
#   IDSSE        +0.017 +0.020 +0.023 +0.024 +0.025 +0.015 +0.006
#   Metrica      +0.007 +0.007 +0.010 +0.006  0.000 -0.028 -0.063
#   SkillCorner  +0.006 +0.008 +0.010 +0.012 +0.016 +0.018 +0.018   (ball as recorded)
# Chosen with each set left out in turn, the other two pick 0.05, 0.1 and 0.05, and the set left
# out then gains +0.023 (IDSSE), 0.000 (Metrica, at 0.1) and +0.010 (SkillCorner): +0.011 on
# average, which is the honest size of this clue. 0.05 is the largest weight at which no set loses.
WIDTH_WEIGHT = 0.05

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

# The control clue (v7, retrained for v8): a classifier trained on the external matches
# (external/train_control.py) says, per frame, how likely each state is given the players and ball
# around it; its posterior over the class shares is added to the emissions at CONTROL_WEIGHT, on
# top of every v6 clue. Replacing the gap-distance and flight clues with it instead was worse on
# all three sets (v7): its windowed features blur the moment of a switch, which those two clues
# mark. Chosen on the external matches with the classifier trained with each set left out
# (external/score_control.py). spellF1 against the control-folded labels,
# at 0.15 / 0.2 / 0.3 (v6: 0.510, 0.764, 0.403):
#   IDSSE        0.636  0.644  0.653
#   Metrica      0.819  0.816  0.817
#   SkillCorner  0.480  0.480  0.476
# The means are 0.645, 0.647 and 0.649, which is flat, and SkillCorner, the set tracked most like
# BAR-ATH, loses at 0.3. 0.2 is the middle. Every weight gains on every set.
CONTROL_MODEL = os.path.join(ROOT, "hmm", "models", "control_state.npz")
CONTROL_WEIGHT = 0.2
CONTROL_FEATURES = "joint1"

# A possession briefer than this with the other team on both sides is given back to that team
# after decoding (fold_brief_possessions). Off by default. On the external matches 1.5 s gains
# 0.005 against the control-folded labels, but those labels are themselves folded by a rule of the
# same shape (external/control.py), so part of that is the yardstick's own construction; against
# the labels as they are it costs Metrica 0.014. And the challenge's ground truth does keep such
# possessions: the public window has two of 17, 0.76 s and 1.32 s long, each between two spells
# of the other team. That was looked at once, to check the assumption, not to choose the number.
MIN_POSSESSION_SECONDS = 0.0

# A possession briefer than this that follows the other team and ends in a stoppage is given back
# to that team after decoding (fold_before_stoppage): the last touch that puts the ball out, or a
# foul won, is not controlled possession. Off by default. On the external matches, against the
# control-folded labels, 1.5 s takes v8's 0.644 / 0.816 / 0.480 to 0.649 / 0.822 / 0.487 and 2 s to
# 0.649 / 0.820 / 0.490, up on all 30 matches (external/post_sweep.py); with the length chosen on
# the other two providers each time the mean gain is 0.003. As with MIN_POSSESSION_SECONDS, the
# folded labels are made by a rule of this shape (external/control.py), so part of the gain is the
# yardstick's construction: against the labels as they are 2 s costs IDSSE 0.007 and Metrica 0.012
# and gains SkillCorner 0.006. The public window has four stoppages; the possessions before them
# are 7.4, 3.6, 3.5 and 7.6 s, so it neither confirms nor contradicts a limit of 2 s or less.
BEFORE_STOPPAGE_SECONDS = 0.0


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


def width_table(home=WIDTH_GIVEN_HOME, dead=WIDTH_GIVEN_DEAD):
    """log p(width category | state), states x width categories. A is H with the bins reversed."""
    home = np.asarray(home) / np.sum(home)  # the rounded constants do not sum to exactly 1
    dead = np.asarray(dead) / np.sum(dead)
    return np.log(np.array([home, home[::-1], dead]))


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

    Every share quoted here is "ball seen": the in-play and outside categories together, live
    row against dead row. (main() prints the five categories; in play alone is 2 points lower in
    each row on BAR-ATH, 83% and 25%.)

    Checked on the external matches, where it is fitted on each match in the same way and the
    labels give the true shares. On IDSSE, where the ball is knocked out at random, it finds 59%
    against 56% (true: 59% and 59%), close to the old hand-set table, and the scores do not move.
    On the Metrica samples it finds 91% against 14% (true: 90% and 3%), and spellF1 rises
    (0.742 -> 0.774). So where the truth is known the live row is right and the dead row errs
    towards neutral. On BAR-ATH it finds 85% against 27%.

    Its known bias. On a frame with the ball in view the gap clue pushes towards live; on a frame
    without it only the players' speed speaks, and at a jog (1.2-1.5 m/s) that leans slightly
    towards dead. So even a ball lost at random comes out a little more seen in the live row:
    3 points on IDSSE, and in the worst synthetic case, everyone jogging through gaps of 10 s, 81%
    against 35% (tests/test_decode.py bounds the ordinary case). How much of BAR-ATH's 85% / 27%
    is that bias is measured by place_placebo: with the ball's missing pattern shifted in time, so
    that it keeps its run lengths but no longer lines up with play, the fit returns 58-62% against
    55-59%.

    What that does and does not show. The placebo shows the table is not made by the fit: on this
    match the ball really is missing where the other clues say dead. It does not show that those
    frames are dead. On a frame without the ball the gap and flight clues are silent, so "dead"
    there means the players are slow and the neighbouring frames agree; if this tracking loses the
    ball in slow live play, the fit will learn that as dead. The one labelled check does not
    settle it: in the public window the ball is seen on 93.5% of the 6,031 live frames and 64.6% of
    the 1,469 dead ones (four stoppages), against 85% and 27% fitted over the match. The window's
    ball is missing on 12% of frames and the match's on 41%, so the dead row must be lower over
    the match than in the window, but how much lower depends on the match's true dead share, which
    we do not have. On the external matches the fitted table, against the hand-set one, is +0.032
    spellF1 on the Metrica samples (3 of 3 up) and -0.004 on IDSSE (1 of 7 up): +0.007 over all
    10, with a 5-95% interval over matches of -0.002 to +0.016. On BAR-ATH it finds 3 fewer of
    the 58 tagged restarts (50 against 53) and 23 blind same-team stoppages against 5. So it is
    not the default; --fit-place decodes with it.
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


def place_placebo(gap, place, speed, flight, log_trans, log_start, segments, kickoffs=KICKOFFS,
                  minutes=(3, 7, 13, 21, 29)):
    """[(minutes, seen given live, seen given dead)]: fit_place with the ball's place shifted by
    that many minutes inside each half, wrapping round. A shifted ball goes missing in runs as
    long as the real one's but at moments unrelated to play, so any gap between the two shares is
    fit_place's bias on this match, not something about its tracking. The gap and flight clues are
    hidden wherever the shifted ball is missing, as they would be. Each half is taken from its
    kickoff, so the frames pinned dead before one are left out. (On a frame where the shifted ball
    is seen and the real one was not, there is no gap to read, so the live push is a little weaker
    than in the real fit and the bias is, if anything, understated.)"""
    play = [(max([lo] + [k for k in kickoffs if lo < k < hi]), hi) for lo, hi in segments]
    out = []
    for shift in minutes:
        moved = place.copy()
        for lo, hi in play:
            moved[lo:hi] = np.roll(place[lo:hi], int(round(shift * 60 * FPS)))
        missing = moved >= MISSING_AFTER_LINE
        others = [(np.where(missing, -1, gap), gap_distance_table()), (speed, speed_table()),
                  (np.where(missing, -1, flight), flight_table())]
        live, dead = fit_place(moved, others, log_trans, log_start, play, kickoffs=())
        out.append((shift, live[:MISSING_AFTER_LINE].sum(), dead[:MISSING_AFTER_LINE].sum()))
    return out


def fold_brief_possessions(path, min_frames):
    """Copy of a state path with every H or A run shorter than min_frames that has the other team
    on both sides given back to that team. Repeats until nothing changes, since folding one run
    joins its neighbours. Runs next to a stoppage are left alone."""
    path = np.array(path, copy=True)
    while min_frames > 1:
        edges = np.flatnonzero(path[1:] != path[:-1]) + 1
        starts, stops = np.r_[0, edges], np.r_[edges, len(path)]
        changed = False
        for i in range(1, len(starts) - 1):
            a, b = starts[i], stops[i]
            before, after = path[a - 1], path[b]
            if (b - a < min_frames and path[a] != D and before == after and before != D
                    and before != path[a]):
                path[a:b] = before
                changed = True
        if not changed:
            break
    return path


def fold_before_stoppage(path, min_frames):
    """Copy of a state path with every H or A run shorter than min_frames that follows the other
    team and ends in a stoppage given back to that team. One pass, on purpose: repeating it would
    fold the run it has just lengthened into the one before, and so on back up a chain."""
    path = np.array(path, copy=True)
    if min_frames <= 1:
        return path
    edges = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(path)]
    for i in range(1, len(starts) - 1):
        a, b = starts[i], stops[i]
        before = path[a - 1]
        if (b - a < min_frames and path[a] != D and path[b] == D and before != D
                and before != path[a]):
            path[a:b] = before
    return path


def decode(clues, log_trans, log_start, segments, kickoffs=KICKOFFS, weight=WEIGHT, extra=(),
           direct=None, min_possession=0, before_stoppage=0):
    """Labels for every video frame: one Viterbi path per half, X everywhere else.

    clues is a list of (obs, log_table), each obs covering every video frame, all at `weight`.
    extra is a list of (obs, log_table, weight) for clues with a weight of their own. direct is a
    (frames, 3) array added to the log emissions as it is (the control clue, already weighted).
    min_possession, in frames, folds briefer possessions between two spells of the other team
    (fold_brief_possessions); 0 leaves the path as Viterbi gives it. before_stoppage, in frames,
    then folds briefer possessions between the other team and a stoppage (fold_before_stoppage)."""
    labels = np.full(len(clues[0][0]), "X", dtype="<U1")
    for lo, hi in segments:
        log_emit = build_log_emit([(obs[lo:hi], table, weight) for obs, table in clues]
                                  + [(obs[lo:hi], table, own) for obs, table, own in extra])
        if direct is not None:
            log_emit = log_emit + direct[lo:hi]
        pin_dead_before_kickoff(log_emit, lo, hi, kickoffs)
        path = fold_brief_possessions(viterbi(log_emit, log_trans, log_start), min_possession)
        labels[lo:hi] = STATES[fold_before_stoppage(path, before_stoppage)]
    return labels


def load_control(game=GAME, data_dir=None, model_path=CONTROL_MODEL):
    """(frames, 3) control clue for every video frame: the classifier's log p(state | features)
    - log p(state) on the graded halves (hmm/control.py), 0 elsewhere. Cached under data/cache,
    keyed on the tracking and on the model file, since evaluating the trees takes a minute."""
    if not os.path.exists(model_path):
        sys.exit("no control model at {}: run external/control_features.py and "
                 "external/train_control.py (needs scikit-learn), or pass --no-control".format(
                     os.path.relpath(model_path, ROOT)))
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)
    keepers = keeper_slots(meta_path)
    halves = graded_segments(game)
    stat = os.stat(model_path)
    # CONTROL_FEATURES names the feature code: change it with joint_features, or a model file
    # that stays the same would be read against scores cached from the old columns
    key = "{}_{}_{}_{}".format(cache_key(raw_path, team_of_track, n_frames, pitch, FPS, keepers,
                                         halves), stat.st_size, int(stat.st_mtime),
                               CONTROL_FEATURES)
    cache_path = os.path.join(CACHE_DIR, "control_{}.npy".format(key))
    if os.path.exists(cache_path):
        return np.load(cache_path)
    print("control clue: reading positions and evaluating the classifier (cached after) ...")
    px, py, ball_x, ball_y = read_positions(raw_path, len(team_of_track), n_frames, pitch)[:4]
    team = team_by_frame(px, team_of_track, keepers, halves, pitch[0])
    model = trees.load(model_path)
    out = np.zeros((n_frames, 3))
    args = (px, py, team, ball_x, ball_y, pitch, FPS, halves)
    X, names, frames = joint_features(*args)
    Xs = joint_features(*args, swap=True)[0]
    missing = set(model["names"]) - set(names)
    if missing:
        sys.exit("the control model needs feature columns hmm/control.py no longer makes: {}"
                 .format(sorted(missing)))
    cols = [list(names).index(n) for n in model["names"]]  # the model's columns, in its order
    X, Xs = X[:, cols], Xs[:, cols]
    inside = np.zeros(n_frames, bool)
    for lo, hi in halves:
        inside[lo:hi] = True
    scored = inside[frames]
    ratio = np.zeros((len(frames), 3))
    ratio[scored] = control_log_ratio(model, X[scored], Xs[scored])
    out[inside] = hold(ratio, n_frames)[inside]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, out)
    return out


def load_observations(game=GAME, data_dir=None):
    """(gap-distance, place, speed, flight, width) categories for every video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    print("reading {} (cached after the first run) ...".format(os.path.basename(raw_path)))
    pitch = GAMES[game]["pitch"]
    keepers = keeper_slots(meta_path)
    ball_x, ball_y, d_home, d_away, speed, width, stats = load_distances(
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
    return gap, place, speed_category(speed), flight, width_category(width)


PLACE_NAMES = ["in play", "outside", "lost@line", "lost@play", "stale"]
SPEED_NAMES = ["<0.6", "<0.9", "<1.2", "<1.5", "<2.0", "<2.5", ">=2.5"]
WIDTH_NAMES = ["<-5", "<-3.5", "<-2", "<-1", "<0", "<1", "<2", "<3.5", "<5", ">=5"]
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
                      "max {:5.0f} s  under 2 s {:.0%}  under 4 s {:.0%}".format(
                          len(spells), name, *(np.array(
                              [np.median(spells), spells.mean(), np.percentile(spells, 90),
                               spells.max()]) / FPS),
                          np.mean(spells < 2 * FPS), np.mean(spells < 4 * FPS)))


def dead_runs(labels, place, segments):
    """Every decoded stoppage inside the halves, as a dict of per-stoppage arrays:

    start, stop     frames, stop exclusive
    same_team       the same team has the ball just before and just after it
    blind           the ball is missing on every frame of it
    into_missing    seconds between the ball going missing and the stoppage starting; NaN if the
                    ball is in view on its first frame

    A blind stoppage with the same team on both sides is where a wrong "missing means dead" would
    show: one possession cut in two by a ball lost in play. It is also what a real stoppage with
    the ball out of view looks like when the same team restarts (a free kick, a goal kick after a
    back-pass), so the count is an upper bound on the splits, not a count of errors."""
    missing = place >= MISSING_AFTER_LINE
    frames = np.arange(len(labels))
    seen_before = np.maximum.accumulate(np.where(missing, -1, frames))  # last frame with the ball
    out = {k: [] for k in ("start", "stop", "same_team", "blind", "into_missing")}
    for lo, hi in segments:
        seg = labels[lo:hi]
        edges = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
        for a, b in zip(edges[:-1], edges[1:]):
            if seg[a] != "D":
                continue
            before = seg[a - 1] if a > 0 else ""
            after = seg[b] if b < len(seg) else ""
            out["start"].append(lo + a)
            out["stop"].append(lo + b)
            out["same_team"].append(before == after and before in ("H", "A"))
            out["blind"].append(bool(missing[lo + a:lo + b].all()))
            out["into_missing"].append((lo + a - seen_before[lo + a] - 1) / FPS
                                       if missing[lo + a] else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def report_dead_runs(labels, place, segments, tags=None):
    """Label-free checks on the decoded stoppages (dead_runs), and against the Smart Tagging set
    pieces if given: Metrica's automatic tags, not ground truth, and they miss about half of a
    match's restarts, so a stoppage without one is not thereby wrong."""
    runs = dead_runs(labels, place, segments)
    n = len(runs["start"])
    split = runs["blind"] & runs["same_team"]
    with np.errstate(invalid="ignore"):
        late = runs["into_missing"] > 2
    print("{} stoppages: {} blind (ball missing throughout), {} same team before and after, "
          "{} both; {} start more than 2 s into a missing ball".format(
              n, int(runs["blind"].sum()), int(runs["same_team"].sum()), int(split.sum()),
              int(late.sum())))
    if not tags:
        return
    graded = np.zeros(len(labels), dtype=bool)
    for lo, hi in segments:
        graded[lo:hi] = True
    pieces = [(t["start_frame"], t["end_frame"]) for t in tags if t["code"].lower() == "set pieces"
              and graded[min(t["start_frame"], len(graded) - 1)]]
    restarts = runs["stop"]  # the first live frame after each stoppage
    found = sum(bool(np.any((restarts >= a - 2 * FPS) & (restarts <= b + 2 * FPS)))
                for a, b in pieces)
    tagged = np.array([any(a - 5 * FPS <= r <= b + 5 * FPS for a, b in pieces) for r in restarts])
    print("Smart Tagging: a decoded restart within 2 s of {} of the {} tagged set pieces. "
          "No tagged set piece within 5 s of their end: {} of all {} stoppages, {} of the {} "
          "that start more than 2 s into a missing ball, {} of the {} blind same-team ones".format(
              found, len(pieces), int(np.sum(~tagged)), n, int(np.sum(late & ~tagged)),
              int(late.sum()), int(np.sum(split & ~tagged)), int(split.sum())))


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
    parser.add_argument("--out", help="default evaluation/results/hmm_v8.csv (hmm_v6.csv with "
                        "--no-control), with the variant flags in the name if any is set")
    parser.add_argument("--no-control", action="store_true",
                        help="ablation: without the control classifier, which is v6")
    parser.add_argument("--control-model", default=CONTROL_MODEL,
                        help="exported classifier (default hmm/models/control_state.npz)")
    parser.add_argument("--control-weight", type=float, default=CONTROL_WEIGHT,
                        help="weight on the control clue (default {})".format(CONTROL_WEIGHT))
    parser.add_argument("--min-possession", type=float, default=MIN_POSSESSION_SECONDS,
                        help="seconds under which a possession between two spells of the other "
                             "team is folded (default {}; 0 for none)".format(
                                 MIN_POSSESSION_SECONDS))
    parser.add_argument("--before-stoppage", type=float, default=BEFORE_STOPPAGE_SECONDS,
                        help="seconds under which a possession between the other team and a "
                             "stoppage is folded (default {}; 0 for none)".format(
                                 BEFORE_STOPPAGE_SECONDS))
    parser.add_argument("--weight", type=float, default=WEIGHT,
                        help="emission weight (v5: {}, v4: {})".format(WEIGHT, WEIGHT_V4))
    parser.add_argument("--hand-gap", action="store_true",
                        help="ablation: v4's hand-set gap table, without the distance")
    parser.add_argument("--no-width", action="store_true",
                        help="ablation: without the width clue, which is v5")
    parser.add_argument("--fit-place", action="store_true",
                        help="count the ball-place table on the match (fit_place) instead of "
                             "using the hand-set one")
    parser.add_argument("--placebo", action="store_true",
                        help="also fit the place table to a ball shifted in time (place_placebo)")
    parser.add_argument("--baum-welch", type=int, nargs="?", const=20, default=0,
                        metavar="N_ITER", help="also run unsupervised EM as a check")
    parser.add_argument("--bw-weight", type=float, default=1.0, metavar="W",
                        help="emission weight for the EM check: 1.0 is plain EM, {} matches "
                             "the decoder (tempered, a heuristic)".format(WEIGHT))
    args = parser.parse_args()

    gap, place, speed, flight, width = load_observations()
    segments = graded_segments()
    report_shares(place, PLACE_NAMES, "ball place", segments)
    report_shares(speed, SPEED_NAMES, "player speed (m/s)", segments)
    report_shares(flight, FLIGHT_NAMES, "flight", segments)
    report_shares(width, WIDTH_NAMES, "width, home minus away (m)", segments)
    log_trans = transitions()
    gap_clue = (gap, gap_distance_table())
    if args.hand_gap:  # the gap bin alone is the category's row in the gap x distance grid
        gap_clue = (np.where(gap >= 0, gap // N_DISTANCE_BINS, -1).astype(np.int8), gap_table())
    others = [gap_clue, (speed, speed_table()), (flight, flight_table())]
    live, dead = PLACE_GIVEN_LIVE, PLACE_GIVEN_DEAD
    if args.fit_place or args.placebo:
        fitted = fit_place(place, others, log_trans, np.log(START), segments, weight=args.weight)
        if args.fit_place:
            live, dead = fitted
    for name, row in (("live", live), ("dead", dead)):
        print("ball place given {}, {}: ".format(name, "fitted" if args.fit_place else "by hand")
              + "  ".join("{} {:.1%}".format(n, p) for n, p in zip(PLACE_NAMES, row)))
    # gap, place, speed, flight: the order report_baum_welch names them in
    clues = [others[0], (place, place_table(live, dead))] + others[1:]

    extra = [] if args.no_width else [(width, width_table(), WIDTH_WEIGHT)]
    direct = None if args.no_control else args.control_weight * load_control(
        model_path=args.control_model)
    fold = 0 if args.no_control else int(round(args.min_possession * FPS))
    labels = decode(clues, log_trans, np.log(START), segments, weight=args.weight, extra=extra,
                    direct=direct, min_possession=fold,
                    before_stoppage=0 if args.no_control else int(round(args.before_stoppage * FPS)))
    ablation = "".join(tag for tag, on in (("_w{:g}".format(args.weight), args.weight != WEIGHT),
                                           ("_cw{:g}".format(args.control_weight),
                                            not args.no_control
                                            and args.control_weight != CONTROL_WEIGHT),
                                           ("_mp{:g}".format(args.min_possession),
                                            not args.no_control
                                            and args.min_possession != MIN_POSSESSION_SECONDS),
                                           ("_bs{:g}".format(args.before_stoppage),
                                            not args.no_control
                                            and args.before_stoppage != BEFORE_STOPPAGE_SECONDS),
                                           ("_handgap", args.hand_gap),
                                           ("_nowidth", args.no_width),
                                           ("_fitplace", args.fit_place)) if on)
    path = write_submission(
        args.out or os.path.join(RESULTS_DIR, "hmm_v{}{}.csv".format(
            6 if args.no_control else 8, ablation)), labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    summarise(labels, segments)
    report_dead_runs(labels, place, segments, load_smart_tagging())

    if args.placebo:
        print("ball seen given live / dead, fitted: {:.1%} / {:.1%}; with the ball shifted "
              "by".format(*(row[:MISSING_AFTER_LINE].sum() for row in fitted)))
        for shift, seen_live, seen_dead in place_placebo(gap, place, speed, flight, log_trans,
                                                         np.log(START), segments):
            print("  {:2d} min: {:.1%} / {:.1%}".format(shift, seen_live, seen_dead))

    if args.baum_welch:
        report_baum_welch(clues, log_trans, segments, args.baum_welch, args.bw_weight)


if __name__ == "__main__":
    main()
