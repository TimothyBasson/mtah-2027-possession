# alldata/physics: possession from the ball itself

One of two independent pushes for raw spellF1 with every data point allowed (the other is
`alldata/learned/`). This one rebuilds possession the way the labels are made: from the ball's
path, the touches in it and who is near it, with dead ball handled on its own, and the
players-only model (`deep/`, bt) as the fallback where the ball is missing. Nothing in `deep/`,
`hmm/` or the players-only BAR-ATH file is changed.

```bash
pip install numpy pandas scipy scikit-learn lightgbm numba kloppy
python alldata/physics/build.py --tag f5 --knock            # feature tables, 30 outside matches
python alldata/physics/build.py --tag f5b --knock --seed 1  # a second simulated copy of each
python alldata/physics/build.py --tag f5 --only bar_ath
python alldata/physics/train.py --run s5h --tag f5 --more f5b --heads            # held out by provider
python alldata/physics/score.py --runs s5h                                       # spellF1, as given and folded
python alldata/physics/train.py --run s5h --tag f5 --more f5b --heads --fold none  # all 30, predicts BAR-ATH
python alldata/physics/run.py --run s5h                                          # evaluation/results/alldata_physics.csv
```

The prior needs the players-only model's saved probabilities under `data/cache/deep/bt/`
(`/mnt/project-files/deep/bt/` in the cloud, or `deep/train.py` and `deep/run.py`).

## What it does

1. **Ball path** (`ball.py`): velocity and velocity change; a touch is a peak of velocity change
   with a player within reach, and takes the nearer team.
2. **Evidence per 0.2 s** (`feats.py`, 216 columns, all home-relative so exchanging the teams is
   an exact symmetry): the ball's place and speed; how near each team is to it now and up to 8 s
   either side; the last three and next three touches, how many in a row each team strings
   together and the touch balance (the organisers' control rule as numbers); whether the ball is
   shown, how long it has been and will be gone, where it was lost and where it comes back; when
   it was last beyond a line; the teams' speeds and shape; the players-only model's probabilities.
3. **Two classifiers** (`train.py --heads`, boosted trees): *dead or alive* reads only where the
   ball is, whether it is shown, the players and the prior; *which team, given alive* reads
   everything. Every match is read twice, as it is and with the teams exchanged.
4. **Decode** (`decode.py`): the same three-state chain and minimum runs as `deep/` (change cost 4,
   1 s minimum), graded frames before each kick-off set to D.
5. Optional: `run.py --tags` draws a stoppage at Smart Tagging set pieces the model plays through
   (`stops/tags.py`).

`rule.py` is the control rule with no learning at all, and `tim.py` runs Tim's decoder on the same
simulated matches so that its labels can be one more input (`train.py --extra tim`).

## BAR-ATH's ball is not the outside matches' ball

Found from the tracking alone, and the reason most of the work went into `common.py`:

- **It is heavily smoothed.** After a kick the ball takes about a second to reach full speed. Its
  acceleration around an isolated kick, which should be a spike, is a bump 1.3 to 1.5 s wide. A
  parabolic moving average of half-width 14 to 18 frames over IDSSE's raw ball reproduces
  BAR-ATH's acceleration quantiles (6.2 / 17.6 / 29.8 m/s² at 50 / 90 / 99% against 5.9 / 17.7 /
  30.5) and its autocorrelation. A touch detector tuned on the raw outside balls finds a fifth of
  the touches on BAR-ATH. The smoothing cannot be undone: the window's spectrum has zeros from
  1 Hz, and a second-order sharpening improved positions (0.46 m to 0.32 m) but not who the touch
  is given to.
- **It goes missing in long gaps only**: 99% of missing frames sit in gaps over 5 s (median 10 s),
  not the 1 to 2 s dropouts `--seen 0.59` makes. The players-only model calls 83% of gap frames
  dead, and 17% of ball-seen frames dead.
- **A dead ball often stays in view**, moving: carried back, rolled to the keeper, placed. IDSSE
  freezes the ball where it went out and the Metrica samples hide it, so only SkillCorner shows a
  real dead ball in view.

So every outside match is first turned into what BAR-ATH's tracker would have given
(`common.shown_ball` mode "barath", `shown_players`): the ball smoothed by that window, hidden in
long gaps over 12% of live play, kept in view through 35% of stoppages on a made-up path from
where it went out to where play restarts, and players lost as BAR-ATH loses them. All scores below
are on matches simulated this way, under seeds the training copies do not share.

What that costs, held out by provider, spellF1 on the labels as given (IDSSE / Metrica /
SkillCorner):

| ball the model is shown | spellF1 |
|---|---|
| as recorded, sharp (IDSSE's frozen ball hidden) | 0.637 / 0.845 / 0.374 |
| smoothed, long gaps | 0.573 / 0.717 / 0.355 |
| ... and players lost, dead ball in view (the simulation used from here on) | 0.566 / 0.683 / 0.393 |

The Metrica samples' sharp ball sits on the players at the instants their event-built labels
change, which flatters any touch clue; BAR-ATH offers nothing like it.

## Results

spellF1, mean of the three providers (each counts once), decode fixed at cost 4 and 1 s minimum
runs. "Folded" is against labels folded to the controlled-possession rule (`external/control.py`).

| held out by provider (trained on the other two) | IDSSE | Metrica | SkillCorner | mean as given | mean folded |
|---|---|---|---|---|---|
| players-only bt (the prior on its own) | 0.582 | 0.651 | 0.402 | 0.545 | 0.626 |
| control rule alone, no learning (`rule.py`) | 0.328 | 0.513 | 0.291 | 0.377 | 0.425 |
| control rule, players-only where the ball is hidden | 0.384 | 0.513 | 0.357 | 0.418 | 0.474 |
| one classifier, three classes (s2) | 0.563 | 0.703 | 0.352 | 0.540 | 0.620 |
| two classifiers, dead ball in view simulated (s3h) | 0.561 | 0.689 | 0.390 | 0.547 | 0.627 |
| ... trained on two simulated copies (**s5h, the model here**) | 0.566 | 0.683 | 0.393 | 0.548 | 0.629 |
| s5h and the prior averaged half and half | | | | 0.559 | 0.642 |

| held out by match, all three providers in training | IDSSE | Metrica | SkillCorner | mean as given | mean folded |
|---|---|---|---|---|---|
| s5h | 0.645 | 0.718 | 0.463 | 0.609 | 0.687 |
| one classifier, three classes (s2) | 0.643 | 0.737 | 0.470 | 0.617 | 0.696 |

Read together:

- **Held out by provider, the ball as BAR-ATH shows it adds almost nothing to the players-only
  model**: +0.003 as given, +0.003 folded, Metrica up 0.03, IDSSE and SkillCorner down. Frame by
  frame the two name the same team equally often (0.888). The players-only network already
  infers who is on the ball from the players, and what the real ball adds is the timing of
  touches, which the smoothing blurs.
- **With the provider in training it adds 0.06**, which is mostly the provider's own labelling
  habits (when IDSSE flips on a first touch, how SkillCorner lags). BAR-ATH's labels are a fourth
  habit nobody can train on.
- **Given the true team the score would be 0.91; given the true stoppages only 0.60** (IDSSE;
  Metrica 0.91 and 0.72). True possessions matched at IoU 0.5 by length on IDSSE: under 2 s 9%,
  2 to 5 s 40%, 5 to 10 s 65%, over 10 s 91%. The loss is short possessions, as in every earlier
  thread.
- **Where it does help is stoppages with the ball in view**, which the outside matches cannot
  score (only SkillCorner has them: 0.352 to 0.390 when they were simulated and the dead-ball
  classifier was kept from reading how the ball moves). On BAR-ATH the model explains all but 2
  of the tagged set pieces with a stoppage of its own, against 4 missed by the players-only
  model.

BAR-ATH's public 5 minutes, looked at once per version as a check and never chosen on (17
possessions; one possession is 0.06): s5h spellF1 0.787, macroF1 0.964, 14 possessions; s3h 0.850;
players-only bt 0.745, macroF1 0.904, 12 possessions. The first version, one classifier with no
dead ball in view in training, scored 0.681 with macroF1 0.807: it played through both stoppages
where the ball stayed in view, which is how that weakness was found.

## Checked in review (`review.py`)

**Is it better than the players-only model? Not measurably.** Paired by match, matches resampled
inside each provider 2000 times, held out by provider:

| | players-only bt | s5h | s5h minus bt (95%) | matches s5h is ahead |
|---|---|---|---|---|
| as given | 0.545 [0.530, 0.559] | 0.548 [0.539, 0.556] | +0.003 [-0.005, +0.010] | 10 of 30 |
| folded | 0.626 [0.615, 0.638] | 0.629 [0.620, 0.638] | +0.003 [-0.003, +0.009] | 10 of 30 |

By provider (as given): IDSSE -0.015 (ahead on 2 of 7), Metrica +0.032 (3 of 3), SkillCorner
-0.009 (5 of 20). The mean is level only because Metrica's three matches count as a third. On
BAR-ATH's public 5 minutes the scorer puts gaps under 0.38 inside noise (17 possessions, 1000
draws), so 0.787 against 0.745 says nothing. **This file is an alternative to the players-only
file, not a replacement for it.**

**Ball in view against ball hidden**, simulated matches: frame accuracy, and true possessions
matched at IoU 0.5 by how much of the possession had the ball in view.

| | accuracy, in view | accuracy, hidden | matched, mostly in view | matched, mostly hidden |
|---|---|---|---|---|
| IDSSE bt / s5h | 0.878 / 0.879 | 0.926 / 0.955 | 0.48 / 0.47 | 0.40 / 0.20 (199) |
| Metrica bt / s5h | 0.893 / 0.907 | 0.767 / 0.792 | 0.59 / 0.62 | 0.00 / 0.05 (19) |
| SkillCorner bt / s5h | 0.805 / 0.794 | 0.834 / 0.850 | 0.40 / 0.42 | 0.29 / 0.25 (619) |

So the gain is not where it was expected. With the ball in view the model names possessions no
better than the players-only model outside Metrica. With the ball hidden it is more accurate
frame by frame because it calls more of the gap dead, and for the same reason it loses
possessions played inside a gap (IDSSE 0.40 to 0.20).

**The BAR-ATH file against the players-only file, no labels** (frames after each kick-off):

| | possessions | mean / median | under 2 s | stoppages | D | runs under 1 s |
|---|---|---|---|---|---|---|
| first half, this file | 122 | 14.2 / 8.3 s | 11% | 43 | 45% | 0 |
| first half, players-only | 126 | 14.1 / 7.5 s | 13% | 46 | 44% | 0 |
| second half, this file | 136 | 12.1 / 7.7 s | 10% | 55 | 46% | 0 |
| second half, players-only | 149 | 11.3 / 6.6 s | 13% | 54 | 45% | 0 |

(D is 45% here and 20% in the public window because the ball is missing on 41% of these frames
and both files call most of a gap dead, while the public window has the ball in view 88% of the
time.) The public window's rate (17 possessions in 5 minutes) would give
about 350 over the match; both files have about 260 to 275, so both merge possessions, this one
slightly more. The public window is the easy part (ball in view 88% against 59% overall).

The two files differ on 6.4% of frames after kick-off: 42% of that is one team against the other
(ball in view on 71% of those frames), 39% is this file dead where players-only is alive (ball
in view on 31%), 19% the reverse (ball in view on 80%). The rate of disagreement is the same
with the ball in view (6.3%) and without (6.6%). The longest disagreements are stretches with no
ball that this file calls dead and players-only gives to a team (24 s at frame 70822, 9 s at
163730), and stoppages with the ball in view (15 s at 18025, in the public window, where this
file is right). Stoppages against the 53 tagged set pieces: this file 98 stoppages, 51 tags
beside one; players-only 100 and 48. Dead share of ball-in-view frames over the whole match:
16% (this file), 17% (players-only); of ball-missing frames 88% and 85%.

Public 5 minutes: D share 20% in the truth, 20% in this file, 16% players-only; frame accuracy
with the ball in view 0.962 against 0.909, without 0.979 against 0.992 (885 frames).

### Where each simulation constant comes from

| constant | value | from |
|---|---|---|
| `SMOOTH_FRAMES` | 14 to 18 | BAR-ATH's tracking only: the ball's acceleration quantiles and autocorrelation |
| `LIVE_GAP_SHARE`, `LIVE_GAP_MEDIAN` | 12%, 7 s | BAR-ATH's tracking (gap lengths), the Metrica samples' own rate of 9 to 13%, and the players-only model's calls on BAR-ATH (17% of gap frames alive). No labels |
| `FROZEN_SECONDS` | 0.6 s | IDSSE's tracking |
| `DEAD_SHOWN_SHARE` | 35% | **The public window's labels were used here.** That BAR-ATH's tracker follows a dead ball was noticed there (two of its four stoppages; 15% of its ball-in-view frames are dead). 35% was set once as a round figure and not swept. It gives a dead share of ball-in-view frames of 14% (IDSSE) and 10% (Metrica) in the simulated matches; SkillCorner's real tracking has 16%, and on the whole of BAR-ATH the players-only model, with no labels, says 17% |
| touch thresholds (`ball.py`), decode (cost 4, 1 s) | | outside matches; decode is `deep/`'s |

No model, threshold or decode setting was chosen on the public window's score.

### Reproducing

Not yet shown from a clean clone. What a rerun needs: the 30 outside matches
(`python external/build.py`, public downloads, see `external/README.md`), the players-only
model's probabilities (`deep/train.py`, `deep/run.py`), then the commands at the top. Versions
used: numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1, lightgbm 4.7.0, numba 0.68.1,
kloppy 3.18.0, joblib 1.6.0. `models/s5h.joblib` is the model that made the BAR-ATH file
(`train.py --fold none` rebuilds it). Retrained from the same feature tables on the same machine,
the probabilities are identical (largest difference 0.0) and the BAR-ATH file is the same byte for
byte; the `.joblib` file itself differs in its pickle bytes. A rebuild of the feature tables on
another machine has not been checked. The Tim-input model (s7h) is not in git.

## With Tim's decoder as an input

`tim.py` reruns Tim's v8 on the same simulated matches (his files untouched, his classifier
refitted held out by provider into this folder's cache) and `train.py --extra tim` gives his
labels to the classifiers as 16 more columns.

| held out by provider | IDSSE | Metrica | SkillCorner | mean as given | mean folded |
|---|---|---|---|---|---|
| Tim's v8 alone on the simulated matches | 0.475 | 0.597 | 0.425 | 0.499 | 0.592 |
| s5h (independent of Tim's model) | 0.566 | 0.683 | 0.393 | 0.548 | 0.629 |
| s7h (s5h plus Tim's labels) | 0.570 | 0.687 | 0.390 | 0.549 | 0.627 |

Held out by match with every provider in training it is 0.615 / 0.692 against 0.609 / 0.687
(+0.005 per match, se 0.001, up on 21 of 30): a small gain that again needs the provider's own
labels in training. Its BAR-ATH file is `alldata_physics_tim.csv` (`tim.py train`, `tim.py
bar_ath`, `train.py --run s7h --extra tim --fold none`, `run.py --run s7h`): 261 possessions,
public 5 minutes 0.747 (check only).

Per match s7h is -0.001 as given (se 0.002, up on 10 of 30) and -0.003 folded: no gain, so the
independent s5h stays the model. His decoder is best on SkillCorner, where this model is
weakest, but the classifier does not turn that into score. Note for anyone who adopts it: his
decoder's shipped tables were set in `hmm/`, and whether any were chosen on the public window
would need checking there first.

## Tried and not kept

| | result |
|---|---|
| Un-smoothing BAR-ATH's ball (spectral, and second-order sharpening) | the window is not invertible; sharpening leaves touch assignment at 0.84 (0.93 on a raw ball) |
| "Carry" features: the ball against players smoothed the same way, relative speed, carry runs | -0.004, 18 of 30 matches up: noise |
| Training on folded labels | folded mean +0.001, as given -0.018; IDSSE and Metrica down on both |
| Averaging with the prior | +0.011 held out by provider, -0.005 to -0.02 with the provider in training: not adopted |
| Decode settings (24 combinations) | 0.531 to 0.548 as given: flat |
| Gating: this model where the ball is in view, the players-only model where it is not (hard switch, and blended over 2 s) | 0.540 and 0.542 as given, 0.619 and 0.621 folded, below both. Leaving the gaps to the players-only model does not get the possessions inside gaps back |
| One three-class classifier | better with the provider in training (+0.008), worse where a dead ball is in view and on BAR-ATH's public window |

## Files

| | |
|---|---|
| `common.py` | loading; the ball and players as BAR-ATH would show them; the prior |
| `ball.py` | ball kinematics and touches |
| `feats.py` | the feature table of one half |
| `build.py` | tables for the 30 matches and BAR-ATH, labels as given and folded |
| `train.py` | the classifiers, held out by provider (`--fold all`), by match (`cv`), or on all 30 (`none`) |
| `decode.py`, `score.py`, `diag.py` | decode, held-out scores, where the loss sits |
| `rule.py` | the control rule with no learning |
| `tim.py` | Tim's decoder rerun on the simulated matches, as an extra input |
| `run.py` | BAR-ATH submission |
| `review.py` | bootstrap against the prior, ball in view against hidden, the BAR-ATH file against the players-only file |
| `models/s5h.joblib` | the classifiers trained on all 30 |
