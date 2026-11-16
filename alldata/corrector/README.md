# Second-stage corrector for brief possessions (not adopted)

Failure mode 1 of the final model: a brief possession (mostly 2 to 5 s) is given to the other
team. This folder leaves the main net alone and tries to fix those after the decode, from ball
physics and the main model's own probabilities. The shipped files are unchanged.

**Result: the corrector removes part of the failure on IDSSE and Metrica, adds about as many
false brief possessions as it finds true ones, and lowers SkillCorner. It does not clear the
bar (no provider down), and both corrected BAR-ATH files score lower on the public five
minutes. Opt-in only.**

## What it does

1. `touch.py` reads touches off the smoothed ball: peaks of its acceleration, each with its
   change of velocity, and the place of the touch put back where it happened (a smoothed ball
   cuts the corner of every kick; the correction needs no knowledge of the filter).
2. `segs.py` makes candidates: inside one decoded possession of team X, a stretch that starts at
   a touch not clearly X's (a player of the other team is the nearer to the ball or to the
   corrected place) and ends at one of the next four touches. One row per candidate: the touches
   (margin between the teams' nearest players, change of velocity, speed in and out, turn),
   one player staying on the ball through the stretch, the main model's probabilities inside
   and around it, where it sits in the decoded possession.
3. `segfit.py` trains boosted trees on those rows with one provider held out. The target is
   whether relabelling the stretch raises the score (matched possessions gained, less the
   possessions added), so the model learns the metric and not only the team. Candidates scored
   above a threshold are relabelled, best first, without overlaps.
4. `train.py` / `run.py`: the same on all IDSSE and Metrica matches, applied to BAR-ATH.

## Held out by provider, BAR-ATH-like balls (base: pw1e, the plain all-data model)

Each head decoded as in `alldata/learned/score.py`. Trained without SkillCorner's labels (its
fold is trained on IDSSE and Metrica in any case). Seeds 0 / 1 where two numbers are given.
"Slice" is the failure itself: the spellF1 that would be gained by making right every true
possession under 5 s that the prediction gives to the other team (mean of providers).

| labels as built | IDSSE | Metrica | SkillCorner | mean | slice |
|---|---|---|---|---|---|
| base | 0.652 | 0.781 | 0.416 | 0.616 | 0.196 |
| corrector, threshold 0.7 | 0.672 / 0.668 | 0.784 / 0.787 | 0.410 / 0.411 | 0.622 / 0.622 | 0.173 / 0.172 |
| corrector, learns the size of the gain | 0.669 / 0.669 | 0.792 / 0.793 | 0.407 / 0.408 | 0.623 / 0.623 | 0.170 / 0.170 |
| every candidate that helps (oracle) | 0.781 | 0.844 | 0.535 | 0.720 | 0.084 |

| folded labels | IDSSE | Metrica | SkillCorner | mean | slice |
|---|---|---|---|---|---|
| base | 0.747 | 0.824 | 0.518 | 0.696 | 0.108 |
| corrector, threshold 0.7 | 0.759 / 0.756 | 0.822 / 0.826 | 0.506 / 0.506 | 0.696 / 0.696 | 0.093 / 0.092 |
| corrector, learns the size of the gain | 0.751 / 0.753 | 0.830 / 0.831 | 0.505 / 0.506 | 0.695 / 0.697 | 0.092 / 0.093 |
| every candidate that helps (oracle) | 0.865 | 0.868 | 0.619 | 0.784 | 0.030 |

In the shipped configuration (the two heads averaged, one decode, one corrector for both
labellings): as built 0.609 to 0.615 / 0.616, folded 0.692 to 0.689 / 0.688.

With SkillCorner's labels in training (a quarter of the weight, size-of-gain version) the gain
on IDSSE disappears (0.656 as built / 0.748 folded) and SkillCorner is still down (0.407 / 0.505).

## BAR-ATH, public five minutes (17 possessions, a check only)

| file | spellF1 | possessions |
|---|---|---|
| alldata_learned_tim.csv (submission) | 0.613 | 15 |
| + corrector (29 stretches relabelled in the match) | 0.590 | 23 |
| alldata_learned.csv | 0.653 | 13 |
| + corrector (26) | 0.567 | 19 |

The window has two brief Home touches (0.8 s and 1.3 s) that merge three Away possessions in
the submission. Of the corrector's four relabelled stretches in the window, two are truly the
other team (the first brief touch, 0.6 s late, and the start of a longer Home possession) and
two are not in the truth.

## What was learned

- The candidates are not the limit: relabelling exactly the ones that help is worth +0.10 as
  built and +0.09 folded. Choosing them is.
- The rule for choosing does not carry across providers. Trained inside a provider (other
  matches of the same source) the model ranks candidates at AUC 0.85 to 0.90; trained on the
  other providers it ranks them below the main model's own probability (IDSSE 0.76 against
  0.79, Metrica 0.81 against 0.87). Even the within-provider version gains only +0.03 on IDSSE
  as built, +0.01 folded, and nothing on the folded mean (first version of the candidates).
- The recorded (unsmoothed) ball does not change this: same ranking, no gain.
- Corrected touch places are not a lever: under BAR-ATH's low-pass the smoothed ball is already
  within 0.3 to 0.6 m of the recorded ball at a touch.
- Re-predicting every frame with boosted trees on per-player holder and touch rows (`cfeats.py`,
  `build.py`, `fit.py`) is worse than the base: 0.598 / 0.674.
- A touch under 2 s is matched only if its timing is right to a few tenths of a second, which
  a ball low-passed at 0.7 Hz does not give.

## Run

    pip install lightgbm numba
    python alldata/corrector/segs.py                 # candidates from data/cache/alldata/runs/pw1e
    python alldata/corrector/segfit.py --sc-share 0.0001 --seeds 0 --thr 0.7        # held-out table
    python alldata/corrector/segfit.py --sc-share 0.0001 --seeds 0 --reg --thr 0.2  # size of the gain
    python alldata/corrector/train.py                # models/segmix_s{0,1}.txt (committed)
    python alldata/learned/run.py && python alldata/corrector/run.py   # evaluation/results/alldata_learned_tim_corr.csv

Not measured: the corrector on held-out predictions of the with-Tim model (never saved; the
corrector is trained on the plain model's and applied to the with-Tim file), and on the
low-pass simulation of the ball (the base probabilities exist for the parabolic one only).
