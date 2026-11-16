# alldata/learned: a learned possession model that reads everything

The players-only rule of `deep/` does not apply in `alldata/`. This model reads the ball as well
as the players, and is judged on spellF1 alone. Nothing in `deep/`, `hmm/` or their BAR-ATH files
is changed. The sibling folder `alldata/physics/` attacks the same goal with rules and physics.

```bash
pip install -r requirements-deep.txt     # or: uv sync --group deep
python alldata/learned/run.py            # BAR-ATH -> evaluation/results/alldata_learned_tim_touch.csv, the submission
python alldata/learned/run.py --no-touch --out alldata_learned_tim.csv     # the previous submission
python alldata/learned/run.py --tags pwN pwNs1 --out alldata_learned_touch.csv   # the fallback, independent of hmm/
python alldata/learned/run.py --tags pwN pwNs1 --no-touch --out alldata_learned.csv
python alldata/learned/run.py --raw-keepers --out alldata_learned_tim_touch_rawk.csv   # keepers as delivered (before PR "keepers")
```

**Keepers (2026-11-16 evening, `possession/keepers.py`).** BAR-ATH's two goalkeeper track slots
wander: the away keeper slot stands in the home goalmouth on 10% of graded frames (a frozen
ghost for 215 s in the second half among them), so the model saw an away player beside every
ball near the home goal. `possession/load.py` now hides frozen tracks and gives a keeper slot
within 25 m of a goal line to the team defending that goal (Tim's `hmm/clues.py` rule, which his
inputs already had). Every model that loads BAR-ATH picks it up; training data is unchanged
(their keepers are named players, never on the wrong team). Submission file with it: sha256
`4979aff0...cb536b6cdf887`; 716 graded frames (0.45%) change, public window level (0.6125 both),
Smart Tagging phase share 0.866 to 0.870. Plain fallback with it `21938e67...364bff442b4c`.
`--raw-keepers` gives back the files below byte for byte.
With this change merged, the default command writes `4979aff0` as the submission in place of
`e9f09d6a`: Gambtt chose it on 2026-11-16 19:43Z ("yes add it and merge it in"), so `4979aff0` is the submission. There is
no measured spellF1 gain; the case is that the input is wrong without it. Validator on the new
file: `OK alldata_learned_tim_touch.csv, 199319 rows, frame 0..199318`. The pseudo-label cache
`data/cache/alldata/bar/BAR-ATH.npz` (only `fit.py --pseudo` reads it) is rewritten by every
`run.py` run with whichever keepers that run used. Moved keeper frames all lie within the penalty-area width (none with |y - 34| > 20 m).

**The submission file is `alldata_learned_tim_touch.csv`** (Gambtt, 2026-11-16 18:34Z: "yes
make it the sub"): the with-Tim model below, decoded with the touches that `alldata/touch/`'s
detector reads off BAR-ATH's ball (`alldata/touch/README.md`). Held out, on the plain all-data
model, the touch evidence takes 0.616 / 0.696 to 0.626 / 0.704 with no provider down; on the
public window it is level (0.613). sha256 `e9f09d6a...776623aba`; the plain fallback with touches
is `7ca261ee...1a0db77cd492`. `--no-touch` gives back the earlier files byte for byte.

Until then the submission file was `alldata_learned_tim.csv` (Gambtt's choice, 2026-11-16): this model
with Tim's `hmm/` v8 as an input, nets `tpsN` and `tpsNs1`. Held out it is the best row measured
on BAR-ATH-like balls (0.623 / 0.705 against 0.616 / 0.696 without his model), with the caveats
in "With Tim's model as an input" below: the gain is on IDSSE only, it is within seed noise, and
his inputs come from simulated balls in training and from the real ball on BAR-ATH. The plain
file `alldata_learned.csv` (nets `pwN`, `pwNs1`) and the players-only file (`deep/run.py`, sw)
are the fallbacks.

One command makes the submission from the provided data alone. `run.py` uses the committed nets
in `models/`, `deep/models/bt.pt`, and for the default nets Tim's v8 on BAR-ATH, which it
produces itself when missing by running his code unchanged (`python -m hmm.decode --no-net`,
then `tim.py --bar`). It reads no ground truth. Checked on 2026-11-16 in a fresh clone holding
only `data/tactical/`: two runs, the second with every cache removed, give the same file byte
for byte (sha256 `75139e82...d6ec5b762`, about 2 minutes on one CPU), and it passes
`evaluation/validate_submission.py` (199,319 rows). The fallback from the same clone is
sha256 `8a3e1bc4...8438db63`, as before. The first file of this folder (`--tags psN psNs1`,
sha256 `0c1ce4a7...c58b935`) is still reproducible from its committed nets.

## The result in one table

Held out by provider (a model never sees the provider it is scored on), spellF1, fixed decode
(cost 4, 1 s minimum possession and stoppage), two seeds averaged. "BAR-ATH-like ball" means the
outside match's ball is hidden and smoothed the way BAR-ATH's is before the model sees it (see
below); that is the row to believe for BAR-ATH.

| ball shown | model | IDSSE | Metrica | SkillCorner | mean, labels as built | mean, folded labels |
|---|---|---|---|---|---|---|
| none | players-only `deep/` bt | 0.582 | 0.651 | 0.402 | 0.545 | 0.641 |
| BAR-ATH-like | first model, trained on balls as recorded (`c1`) | 0.486 | 0.592 | 0.401 | 0.493 | 0.601 |
| BAR-ATH-like | trained on BAR-ATH-like balls (`s1`, `s1s1`) | 0.627 | 0.758 | 0.409 | 0.598 | 0.688 |
| BAR-ATH-like | that, adapted to the unseen provider without labels (`ps1`, `ps1s1`) | 0.631 | 0.759 | 0.416 | 0.602 | 0.692 |
| BAR-ATH-like | SkillCorner drawn for a quarter of training windows (`w1`, `w1s1`) | 0.648 | 0.781 | 0.409 | 0.613 | 0.688 |
| BAR-ATH-like | **that, adapted (`pw1`, `pw1s1`)** | 0.652 | 0.781 | 0.416 | **0.616** | **0.696** |
| BAR-ATH-like | Tim's `hmm/` v8 (`bench.py 1`) | 0.475 | 0.612 | 0.425 | 0.504 | 0.590 |
| BAR-ATH-like | with Tim's v8 as an input (`ts1`, `ts1s1`) | 0.680 | 0.761 | 0.405 | 0.615 | 0.700 |
| BAR-ATH-like | with Tim's v8 as an input, adapted (`tps1`, `tps1s1`) | 0.687 | 0.767 | 0.416 | 0.623 | 0.705 |
| as recorded | Tim's `hmm/` v8 (`bench.py`) | 0.666 | 0.760 | 0.420 | 0.615 | 0.673 |
| as recorded | trained on balls as recorded (`c1`, `c1s1`) | 0.679 | 0.832 | 0.394 | 0.635 | 0.707 |
| as recorded | that, adapted (`p1`, `p1s1`) | 0.698 | 0.860 | 0.401 | 0.653 | 0.718 |
| as recorded | with Tim's v8 as an input (`t1`, `t1s1`) | 0.740 | 0.839 | 0.402 | 0.661 | 0.724 |
| as recorded | with Tim's v8 as an input, adapted (`tp1`, `tp1s1`) | 0.752 | 0.859 | 0.412 | 0.674 | 0.732 |

The model that labels BAR-ATH is the bold row: +0.071 over the players-only model on the labels
as built and +0.055 on the folded labels, ahead on all three providers and on 26 to 27 of the 30
matches (standard error of the paired difference 0.006 to 0.008). Seed to seed the mean moves by
up to 0.01.

**SkillCorner's share.** The thread "Clean up noisy training labels" (PR 42) found SkillCorner's
labels to be the noisy ones and gained by drawing its matches for only a quarter of the training
windows. The same here (`fit.py --share skillcorner=0.25`, `final_share.sh`): against the row
above it, IDSSE +0.021 / +0.009 and Metrica +0.022 / +0.002, 9 of those 10 matches up. The
SkillCorner fold has no SkillCorner in training, so it cannot move and its 20 matches are ties;
the mean gain is +0.014 / +0.004. For the BAR-ATH nets, trained on all three providers, the share
goes from a third to a quarter.

How much to read into it: 0.25 is the only share tried here. It was taken from PR 42, which
tried 0.25 and 0.10 on the players-only net on these same folds, so the choice is not fully
independent of the matches it is scored on. The gain on labels as built is about two standard
errors and on the folded labels it is inside seed noise. Read it as no worse and probably a
little better; the reason to ship it is PR 42's finding about SkillCorner's labels.

The shipped file averages the model's two output heads (below), which scores 0.609 / 0.692
(0.599 / 0.687 before the share change).

## What it is

- **Inputs** (`feats.py`), one row every 0.2 s: the ball (seen or not, how long unseen, position
  carried across gaps, velocity, change of velocity, distance to the lines); each team against
  the ball (its three nearest players, closing speed, players within 2 m and 5 m); touches read
  off the ball's velocity (who touched last and who touches next, how long ago, how many in a row
  by one team); each team alone (players tracked, centroid); and the players-only net's
  probabilities (`deep/` bt). Swapping the teams and mirroring the pitch are exact symmetries of
  the rows and are used in training and averaged over at prediction.
- **Network** (`fit.py`): a small temporal network (12 dilated convolutions, a two-way GRU, 96
  wide) over those rows. About 17 minutes a fold on one CPU.
- **Two heads**, taught together: the labels as each provider built them, and the same labels
  folded to the controlled-possession rule (`external/control.py`). Held out, each head is best
  on its own labelling (0.602 / 0.679 and 0.593 / 0.692); their average is within 0.005 of the
  better one on both (0.599 / 0.687, `heads.py`). Which labelling BAR-ATH's truth is closer to is
  not known (its public window has possessions of 0.8 s and 1.3 s, which the folded rule would
  remove), so the file uses the average.
- **Adaptation without labels** (`fit.py --pseudo`): the unseen matches join the training set
  labelled by the first model's own decoded predictions, leaving out frames where it is under
  0.7 sure. Held out this is done on the held-out provider's matches without their labels; for
  the file it is done on BAR-ATH. RULES.md allows pseudo-labels made by an algorithm on the
  provided game. Worth +0.004 on BAR-ATH-like balls and +0.012 to +0.018 on balls as recorded.
- **The players-only net as an input**: for every outside match its probabilities come from the
  fold where that match's provider was held out, so they are as good as BAR-ATH's will be.
  Without them the first model loses 0.002 to 0.004 (`b3`), so they matter little once the ball
  is in.

## Why a ball-fed model transfers here when the teacher did not

The ball-fed teacher of "Retrain model for short possessions" (a set-attention net over players
with the ball appended) scored 0.50 on a held-out provider, under the players-only model. This
model scores 0.635 on the same question with the same ball. The difference is the input:
relations between the ball and the nearest players, not raw positions, which leaves a provider's
tracking style little room to be memorised. On unseen matches of providers it trained on it
scores 0.73 / 0.78, so about 0.1 of provider difference remains, most of it in how the providers
cut possessions (IDSSE flips on a first touch, and SkillCorner's labels are noisy, as that thread
found).

## BAR-ATH's ball is not like the outside balls

Two properties of BAR-ATH's ball decide whether any of this carries over. Both are measured on
its tracking alone, without labels.

1. **It is smoothed.** A kick takes about a second to reach full speed: the 99th percentile of
   its acceleration is 31 m/s2, against 265 on the Metrica samples and 237 on IDSSE. The
   `alldata/physics` thread found this; checked here, a parabolic window of 14 to 18 frames'
   half-width over a raw outside ball gives 24 to 29 m/s2. A model trained on raw balls reads
   touches off sharp changes of velocity that BAR-ATH never shows: `c1`, 0.635 on balls as
   recorded, scores 0.493 on BAR-ATH-like balls, below the players-only model.
2. **It is missing in long stretches, not at random.** About 75 gaps a half, median 10 s, 99% of
   the unseen time in gaps over 5 s. It is tracked through live play and lost for the core of
   most stoppages. The outside sources each have their own habit (the Metrica samples hide the
   ball on 92% of dead frames, IDSSE parks it where it went out, SkillCorner shows it half the
   time), so "unseen means dead" would be learned as a provider's habit.

`feats.bar_like` therefore rewrites every outside ball before training and before scoring:
parked balls are hidden, each stoppage keeps its ball for a random few seconds at either end and
loses it between (a share of stoppages keep it throughout), a few gaps are cut into live play,
and the path is smoothed. After it, the outside features have BAR-ATH's distributions (velocity
change 0.15 +- 0.18 against BAR-ATH's 0.16 +- 0.19; before, +- 0.28).

**A caveat that matters.** The simulation places the hidden stretches with the labels, on the
scored matches too, so part of the measured gain could be the model reading stoppages off the
gaps more easily than BAR-ATH will let it. `heads.py --always-seen` tests the other
extreme: the same nets on the same matches with the ball in view through every stoppage, so the
gaps say nothing. One seed (`s1`), each head on its own labelling: 0.587 / 0.678 against
0.594 / 0.685 with the simulated gaps, and 0.545 / 0.641 for the players-only model. So at most
0.007 of the gain rests on where the gaps fall; alive/dead F1 goes from 0.960 to 0.956
(players-only 0.940). What the simulation cannot vouch for is the smoothing itself: it matches
BAR-ATH's statistics, not its tracker.

## How far the simulation can be trusted

The held-out numbers are on outside matches with a simulated BAR-ATH ball, not on BAR-ATH. Three
checks of the simulation itself (`heads.py`, `lagcheck.py`; one seed, `s1`, each head on its own
labelling; the reference is 0.594 / 0.685, players-only 0.545 / 0.641):

| the ball the nets are shown | as built | folded |
|---|---|---|
| as trained (window half-width 14 to 18 frames) | 0.594 | 0.685 |
| visible through every stoppage (`--always-seen`) | 0.587 | 0.678 |
| less smoothing, half-width 8 (`hw=8`) | 0.622 | 0.709 |
| more smoothing, half-width 24 (`hw=24`) | 0.542 | 0.649 |
| 0.32 s late (`shift=8`) | 0.557 | 0.655 |
| 0.32 s early (`shift=-8`) | 0.557 | 0.651 |
| 0.48 s late (`shift=12`) | 0.509 | 0.611 |

So the gain survives the gaps saying nothing, and a ball a third of a second early or late
(still ahead of players-only), but not a ball half a second late or one smoothed half as much
again. Does BAR-ATH's ball do either? `lagcheck.py` measures it without labels: at each sharp
turn of the ball's path, when is the nearest player closest to the ball?

| | turns | closest approach, median | quartiles | velocity change at the turn, median / 90th pct |
|---|---|---|---|---|
| BAR-ATH | 1,065 | -0.08 s | -0.32 / +0.32 s | 2.9 / 4.6 m/s |
| Metrica samples, BAR-ATH-like | 934, 1,137 | 0.00, -0.04 s | -0.44 / +0.48 s | 2.9 / 4.5 m/s |
| IDSSE, BAR-ATH-like | 1,107, 590 | -0.24, -0.28 s | -0.6 / +0.36 s | 2.8 / 4.1 m/s |
| Metrica samples, as recorded | 1,592, 1,778 | 0.00 s | -0.04 / +0.1 s | 11.9 / 23.1 m/s |

BAR-ATH's ball is not late against its players (under 0.1 s), and its turns are as sharp as the
simulated Metrica balls' and a little tighter in time.

The smoothing width, from the ball's acceleration alone (moving ball; 90th and 99th percentile,
and for how many frames the acceleration stays correlated above 0.5):

| | 90th pct | 99th pct | correlated for |
|---|---|---|---|
| BAR-ATH, half 1 / half 2 | 18.4 / 19.4 m/s2 | 31 / 48 m/s2 | 11 / 9 frames |
| outside ball, half-width 12 (Metrica / IDSSE) | 20.3 / 19.5 | 38 / 33 | 8 / 9 |
| half-width 14 | 18.1 / 17.5 | 33 / 28 | 10 / 10 |
| half-width 16 | 16.4 / 15.8 | 29 / 25 | 11 / 11 |
| half-width 18 | 15.1 / 14.4 | 26 / 22 | 12 / 12 |
| half-width 24 | 11.9 / 11.3 | 19 / 17 | 14 / 15 |

BAR-ATH sits at a half-width of 12 to 16: at or under the 14 to 18 the nets were trained on,
which is the side where the score rises (half-width 8: 0.622), and far from 24, where it is
lost. Neither failure case of the first table is what BAR-ATH shows. The shifted rows are one
seed and only 0.01 above players-only, so "survives a third of a second" means level, not
ahead. All of this is label-free evidence about the simulation; it is still not a score on
BAR-ATH.

**Is it slow to hand the ball over?** The first hand-over of the public window is 5 s late (see
below), and the BAR-ATH file has fewer possessions in half 1 than the other files. Held out, on
the folded labels, with the file's averaged heads (IDSSE / Metrica / SkillCorner): predicted
over true possessions 0.90 / 0.98 / 1.23 (players-only 0.94 / 0.91 / 1.30); median onset of a
matched possession -0.05 / +0.12 / 0.00 s (players-only -0.21 / +0.31 / -0.32 s); onsets more
than 1 s late 10% / 9% / 21% (players-only 10% / 16% / 21%). So on outside matches it neither
under-counts on the Metrica samples nor starts possessions late. Mixing its probabilities with
the players-only net's (25% or 50%) changes nothing (0.597 / 0.687 and 0.594 / 0.686 against
0.599 / 0.687).

## BAR-ATH

- File: `evaluation/results/alldata_learned.csv` (also `/mnt/project-files/alldata/learned/`).
  Without labels (`barath_report.py`), per half:

  | file | possessions | median length | under 2 s | per live minute | dead | stoppages |
  |---|---|---|---|---|---|---|
  | this model | 108 / 141 | 9.4 / 7.1 s | 6% / 10% | 3.8 / 5.2 | 47% / 49% | 52 / 64 |
  | this model before the share change (`psN`) | 104 / 140 | 10.2 / 6.8 s | 6% / 11% | 3.7 / 5.2 | 47% / 49% | 49 / 62 |
  | with Tim's v8 as an input | 131 / 150 | 7.6 / 7.4 s | 13% / 10% | 4.5 / 5.3 | 45% / 46% | 53 / 65 |
  | players-only `deep` bt | 126 / 149 | 7.5 / 6.6 s | 13% / 13% | 4.3 / 5.3 | 44% / 47% | 46 / 55 |
  | `hmm_v8` | 127 / 137 | 8.0 / 7.8 s | 10% / 10% | 4.4 / 5.0 | 46% / 48% | 50 / 60 |

  The public window has 4.2 possessions per live minute. The file agrees with the one before
  the share change on 98% of graded frames. The rest of this section was measured on that
  earlier file (`psN`); the public-window line is given for both. That earlier model agrees with the players-only
  file on 91.7% of graded frames where the ball is seen and 91.4% where it is not, and with
  `hmm_v8` on 88.7% and 93.6%. All three call 85 to 89% of ball-unseen frames dead and 17 to 19%
  of ball-seen frames dead.
- The two heads on BAR-ATH (`run.py --head built|folded`): 110 / 153 possessions from the head
  taught on labels as built, 102 / 136 from the folded head, 104 / 140 from their average; each
  agrees with the average on over 99% of frames, so the choice of head moves little here.
- Public 5 minutes, a check and not a yardstick (17 possessions): spellF1 0.653 (bootstrap
  interval 0.45 to 0.83), macroF1 0.905; before the share change 0.648 (0.46 to 0.82) and 0.913;
  the with-Tim file 0.613 (0.44 to 0.79) and 0.846; the players-only file 0.745 (0.52 to 0.90) and 0.904;
  `hmm_v8` 0.679 (0.50 to 0.85) and 0.861. Possession by possession the whole gap to the
  players-only file is the first hand-over of the match: the truth gives the ball to Barcelona
  at frame 11,351, this model 5.4 s later and `hmm_v8` 4.9 s later, the players-only net 0.2 s
  later. In those seconds the tracked ball lies 1 to 5 m from Athletic players and 6 to 9 m from
  Barcelona's, so both ball-reading models wait for a Barcelona touch. That costs two matched
  possessions (the 6.6 s and the 8.1 s). On the other 19 stretches this model is level with the
  players-only file or ahead: it finds part of the 14.5 s stoppage at frame 18,027 that the
  players-only file plays through. All three files miss the 0.8 s, 1.3 s and 3.5 s possessions.
  Nothing was chosen on the window.

## With Tim's model as an input ("explore both")

`tim.py` runs `hmm/` v8 unchanged and hands its decoded labels and its control classifier's
posteriors to the network as six extra columns; held out, his classifier is the one trained
without the match's provider (his own protocol). Nothing of his is modified.

On balls as recorded this is the best row of the table (0.674 / 0.732 adapted). On BAR-ATH-like
balls it had to be measured again, because his model reads the ball too: `tim_barlike.py` builds
his features on the same smoothed, hidden balls (versions 1 to 3), refits his held-out
classifiers and decodes, and `final_tim.sh` trains on those inputs.

- His v8 alone drops from 0.615 / 0.673 to 0.504 / 0.590 on BAR-ATH-like balls, under the
  players-only model (the physics thread measured 0.499 / 0.592).
- As an input it still helps: 0.623 / 0.705 adapted, against 0.602 / 0.692 without it and the
  same sampling, and 0.616 / 0.696 for the shipped model. All of that is IDSSE (0.687 against
  0.631 and 0.652). Metrica, the provider whose tracking BAR-ATH's is, goes 0.767 against 0.759
  and 0.781, and SkillCorner is level. With the file's averaged heads: 0.617 / 0.696 against
  0.609 / 0.692.
- On BAR-ATH his inputs come from his own file on the real ball (`hmm_v8.csv`), not from a
  simulation. The with-Tim file has more and shorter possessions (131 / 150 a half) and agrees
  with the shipped file on 95% of graded frames.
- A mismatch the held-out rows do not measure: in training and held out his outputs come from
  simulated balls, where his model is weak (0.504 / 0.590), and on BAR-ATH they come from the
  real ball, where it may be stronger. The net's trust in those columns is set for the weaker
  case, so 0.623 / 0.705 does not carry over to BAR-ATH the way the shipped model's number is
  meant to.

So the with-Tim file (`alldata_learned_tim.csv`, nets `tpsN`, `tpsNs1`) is offered as a second
file, not the main one: its mean gain is under 0.01 and comes from one provider, and it makes
the result depend on his pipeline. `final_both.sh` trains the two changes together (SkillCorner's
share and his inputs); it has not finished, no result or net of it is in this folder.

## Tried and not kept

| | result |
|---|---|
| 3,000 steps instead of 1,500 (`b2`) | +0.001 / +0.002 |
| Training on the folded labels only (`a1f`) | 0.587 / 0.695 against 0.623 / 0.691: each labelling is best learned by its own head, hence two heads |
| Dropping the players-only net's probabilities (`b3`) | -0.002 / -0.004 |
| Decode settings picked on validation matches | the shared setting is within 0.007 of the best on labels as built and 0.002 on folded |
| Averaging two seeds | +0.002 |

Not tried: Smart Tagging and the homography (no outside match has them, so nothing held out can
score them; `deep/run.py --tags` remains the tested way to use set-piece tags); a set-attention
net over all players with ball-relative inputs; finer time steps than 0.2 s; SoccerTrack as a
fourth provider (it has no held-out players-only probabilities).

## Where it loses, and what blending adds (2026-11-16)

`diag.py --tag pw1e` on the shipped recipe's held-out folds (BAR-ATH-like balls, two seeds
averaged; IDSSE / Metrica / SkillCorner / mean):

| what the prediction is given | labels as built | folded labels |
|---|---|---|
| nothing (the model) | 0.652 / 0.781 / 0.416 / 0.616 | 0.747 / 0.824 / 0.518 / 0.696 |
| the true stoppages | 0.655 / 0.763 / 0.455 / 0.624 | 0.750 / 0.803 / 0.552 / 0.702 |
| the true possessions, the model naming each (a tie takes its nearest team) | 0.707 / 0.848 / 0.638 / 0.731 | 0.814 / 0.902 / 0.731 / 0.816 |
| the true team on the frames it calls live | 0.963 / 0.934 / 0.874 / 0.924 | 0.959 / 0.930 / 0.859 / 0.916 |

These are outside matches with a simulated BAR-ATH-like ball: they show where the model is weak
in kind, not what a fix would gain on BAR-ATH.

Stoppages are no longer where the score is: perfect ones add under 0.01. The loss is turnovers.
True possessions found at IoU 0.5 by length, labels as built (IDSSE / Metrica / SkillCorner):
under 2 s 16% / 28% / 5%, 2 to 5 s 54% / 78% / 31%, 5 to 10 s 78% / 89% / 51%, 10 to 20 s
91% / 92% / 64%. Of the missed ones 46 to 49% carry the wrong team for most of their length and
27 to 40% sit inside a longer predicted possession of their own team; edges and fragmentation
are under 7% outside SkillCorner. One Metrica oddity: 25 of its 30 missed possessions over 20 s
are mostly called dead. Those are stretches labelled live in which the ball as recorded is
missing (60 to 100% of each, up to 130 s on end). The Metrica samples' labels come from events,
live until the next ball-out, foul or shot-out, so a stoppage without such an event stays live.
The players-only net, which never sees the ball, calls 59% of Metrica's live frames with a
hidden ball dead as well (this model 67%; IDSSE 11% and 16%), and handing hidden-ball frames to
the players-only net changes nothing at half weight (0.617 / 0.696) and loses at full weight
(0.608 / 0.689). So this is mostly Metrica's labelling, not a decode fault.

**Late blends with the other families add nothing** (`blend.py`). Every family read the same
simulated ball (version 1): Tim's v8 through `tim_barlike.py`, the physics model
(`alldata/physics`, PR 41) rebuilt and retrained held out by provider on balls dumped from
`build.ball_version` (versions 1 and 2). Weighted mean of probabilities, weights fixed in
advance, same decodes:

Mean of the three providers, then the paired difference to this model per match (standard error;
matches up of 30):

| | labels as built | folded labels |
|---|---|---|
| this model (`pw1e`) | 0.616 | 0.696 |
| physics alone | 0.547, -0.044 (0.007; 1) | 0.641, -0.042 (0.005; 1) |
| Tim's v8 labels alone | 0.501, -0.053 (0.017; 14) | 0.594, -0.060 (0.013; 7) |
| + physics at 0.15 | 0.615, -0.001 (0.001; 12) | 0.697, -0.001 (0.001; 13) |
| + physics at 0.3 | 0.612, -0.002 (0.001; 13) | 0.696, -0.001 (0.001; 11) |
| + physics at 0.5 | 0.604, -0.005 (0.002; 12) | 0.692, -0.002 (0.002; 14) |
| + Tim's v8 labels at 0.15 | 0.613, -0.000 (0.001; 14) | 0.697, +0.003 (0.001; 20) |
| + Tim's v8 labels at 0.3 | 0.607, -0.002 (0.003; 17) | 0.693, +0.002 (0.003; 22) |
| + Tim's v8 labels at 0.5 | 0.564, -0.009 (0.012; 19) | 0.667, -0.004 (0.008; 18) |
| + Tim's classifier posteriors at 0.15 | 0.615, +0.000 (0.001; 14) | 0.696, +0.001 (0.001; 19) |
| + Tim's classifier posteriors at 0.3 | 0.610, -0.001 (0.002; 18) | 0.695, +0.002 (0.002; 20) |
| + Tim's classifier posteriors at 0.5 | 0.595, -0.004 (0.006; 19) | 0.691, +0.003 (0.004; 19) |
| + both at 0.15 | 0.613, -0.001 (0.001; 13) | 0.696, +0.001 (0.001; 16) |
| + both at 0.3 | 0.610, -0.002 (0.002; 15) | 0.695, +0.001 (0.002; 18) |
| + both at 0.5 | 0.599, -0.006 (0.004; 13) | 0.690, -0.000 (0.003; 16) |

(The first number weights the providers equally, the difference weights the 30 matches equally,
20 of them SkillCorner, so the two need not agree in the last digit.)

No row clears +0.01, and none is up on every provider under both labellings. Tim's model fed to
the network as an input (above, +0.007 / +0.009, all of it on IDSSE and itself within seed
noise of 0.01) remains the only form of combination that moves the mean at all. Why (`union.py`): of the true possessions this model misses, Tim's v8 finds 6 to 13%,
the physics model 6 to 8%, the players-only net 7 to 12%, and any of the three 16 to 18%. The
families miss the same turnovers, so there is little for a blend to recover.

How the physics rows were made (its code is not on main): branch `claude/project-thread-tkn4qq`
(PR 41) at commit `1700591`, with `phys_shared_ball.patch` of this folder applied, which adds a
ball mode that loads this folder's simulated balls instead of drawing its own. From that
checkout, sharing this checkout's `data/`:

```bash
# here: dump build.ball_version(d, name, v) for v = 1, 2 to data/cache/alldata/learned_ball/<match>.v<v>.npy
python alldata/physics/build.py --mode learned --tag fL --knock
python alldata/physics/build.py --mode learned --tag fLb --knock --seed 1
python alldata/physics/train.py --run sLh --mode learned --tag fL --more fLb --heads   # -> data/cache/alldata/pred/sLh/
```

Tim's rows: `tim_barlike.py features` and `decode` (version 1 is enough). Then, from the
repository root, `blend.py --base pw1e --other phys:sLh` (or `timlab`, `tim`, or several) and
`union.py`, whose run tags are fixed in the script.

## Reproduce

```bash
python external/control_features.py && python external/train_control.py --loso  # only for the with-Tim columns
python alldata/learned/tim.py && python -m hmm.decode && python alldata/learned/tim.py --bar
python alldata/learned/build.py          # feature cache, 30 matches, about 3 minutes
python alldata/learned/run.py --features # BAR-ATH's rows
sh alldata/learned/final.sh              # held-out folds and the BAR-ATH nets, about 1.5 h on 4 CPUs
sh alldata/learned/final_share.sh        # the same with SkillCorner's share at a quarter: the shipped nets
python alldata/learned/tim_barlike.py features && python alldata/learned/tim_barlike.py decode
python alldata/learned/build.py && sh alldata/learned/final_tim.sh   # the with-Tim nets
python alldata/learned/score.py --tags deep:bt ps1e pw1e tps1e --key s
python alldata/learned/run.py
```

Training has not been repeated from a clean clone; the files come from the committed nets, and
two runs of `run.py` agree byte for byte.

`build.py` needs the players-only probabilities of `deep/` (tags `bt`, per fold) under
`data/cache/deep/bt/`; they are in `/mnt/project-files/deep/bt/`. Tags: `a*`, `b*`, `c*`, `p*`,
`t*`, `tp*` were trained on balls as recorded (the feature cache of that time; rebuild with
`feats.bar_like` bypassed to repeat them), `s*`, `ps*`, `w*`, `pw*`, `ts*`, `tps*` on BAR-ATH-like balls; a trailing `N`
is the net trained on all 30 matches, `s1` a second seed, `e` an average of seeds (`ens.py`).
