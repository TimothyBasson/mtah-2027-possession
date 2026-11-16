# alldata/learned: a learned possession model that reads everything

The players-only rule of `deep/` does not apply in `alldata/`. This model reads the ball as well
as the players, and is judged on spellF1 alone. Nothing in `deep/`, `hmm/` or their BAR-ATH files
is changed. The sibling folder `alldata/physics/` attacks the same goal with rules and physics.

```bash
pip install -r requirements-deep.txt
python alldata/learned/run.py            # BAR-ATH -> evaluation/results/alldata_learned.csv
```

`run.py` uses the two committed nets in `models/` (`psN`, `psNs1`) and `deep/models/bt.pt`. Two
runs give the same file (sha256 `0c1ce4a7...c58b935`), and it passes
`evaluation/validate_submission.py`. It reads no ground truth.

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
| BAR-ATH-like | **that, adapted to the unseen provider without labels (`ps1`, `ps1s1`)** | 0.631 | 0.759 | 0.416 | **0.602** | **0.692** |
| as recorded | Tim's `hmm/` v8 (`bench.py`) | 0.666 | 0.760 | 0.420 | 0.615 | 0.673 |
| as recorded | trained on balls as recorded (`c1`, `c1s1`) | 0.679 | 0.832 | 0.394 | 0.635 | 0.707 |
| as recorded | that, adapted (`p1`, `p1s1`) | 0.698 | 0.860 | 0.401 | 0.653 | 0.718 |
| as recorded | with Tim's v8 as an input (`t1`, `t1s1`) | 0.740 | 0.839 | 0.402 | 0.661 | 0.724 |
| as recorded | with Tim's v8 as an input, adapted (`tp1`, `tp1s1`) | 0.752 | 0.859 | 0.412 | 0.674 | 0.732 |

The model that labels BAR-ATH is the bold row: +0.057 over the players-only model on the labels
as built and +0.051 on the folded labels, ahead on all three providers and on 26 to 27 of the 30
matches (standard error of the paired difference 0.006). Seed to seed the mean moves by up to
0.01.

The shipped file averages the model's two output heads (below), which scores 0.599 / 0.687.

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

## BAR-ATH

- File: `evaluation/results/alldata_learned.csv` (also `/mnt/project-files/alldata/learned/`):
  244 possessions of mean 13.6 s, 48% of graded frames dead. It agrees with the players-only
  file on 91.5% of graded frames and with `hmm_v8.csv` on 90.7%.
- Public 5 minutes, a check and not a yardstick (17 possessions, noise about +-0.2): spellF1
  0.648, macroF1 0.913; the players-only file 0.745 and 0.904; `hmm_v8` 0.679 and 0.861. The
  gap to the players-only file is two possessions. One is worth knowing: in the second
  possession of the match the truth gives the ball to Barcelona for 8 s while the tracked ball
  sits 1 to 5 m from Athletic players and 6 to 9 m from Barcelona's; the players-only net calls
  it for Barcelona, this model waits for a Barcelona touch. Nothing was chosen on the window.

## With Tim's model as an input ("explore both")

`tim.py` runs `hmm/` v8 unchanged and hands its decoded labels and its control classifier's
posteriors to the network as six extra columns; held out, his classifier is the one trained
without the match's provider (his own protocol). On balls as recorded this is the best row of
the table (0.674 / 0.732 adapted), almost all of it from IDSSE (+0.05 to +0.06); Metrica, the
provider closest to BAR-ATH, does not move. It has **not** been re-measured on BAR-ATH-like
balls: his classifier's held-out models are not saved, and re-fitting them on smoothed balls
takes about an hour. The physics thread measured v8 itself on its BAR-ATH-like matches at 0.499
/ 0.592 and found no gain from it as an input. So no with-Tim file is offered for BAR-ATH yet;
the independent model is the one shipped. `fit.py` without `--drop-cols tim` trains the variant.

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

## Reproduce

```bash
python external/control_features.py && python external/train_control.py --loso  # only for the with-Tim columns
python alldata/learned/tim.py && python -m hmm.decode && python alldata/learned/tim.py --bar
python alldata/learned/build.py          # feature cache, 30 matches, about 3 minutes
python alldata/learned/run.py --features # BAR-ATH's rows
sh alldata/learned/final.sh              # held-out folds and the BAR-ATH nets, about 1.5 h on 4 CPUs
python alldata/learned/score.py --tags deep:bt s1e ps1e --key s
python alldata/learned/run.py
```

`build.py` needs the players-only probabilities of `deep/` (tags `bt`, per fold) under
`data/cache/deep/bt/`; they are in `/mnt/project-files/deep/bt/`. Tags: `a*`, `b*`, `c*`, `p*`,
`t*`, `tp*` were trained on balls as recorded (the feature cache of that time; rebuild with
`feats.bar_like` bypassed to repeat them), `s*` and `ps*` on BAR-ATH-like balls; a trailing `N`
is the net trained on all 30 matches, `s1` a second seed, `e` an average of seeds (`ens.py`).
