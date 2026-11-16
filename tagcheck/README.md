# A BAR-ATH check from the Smart Tagging

The public five minutes hold 17 possessions, so spellF1 there moves by about ±0.19 from the draw
alone and it has disagreed with the held-out scores. This folder asks whether BAR-ATH's Smart
Tagging, which covers the whole match, gives a steadier way to choose between submission files.

**Answer: it is a good smoke test and a poor referee.** It separates a broken or baseline file
from a real model at once, over the whole match. It cannot rank the real candidates: they sit
within 0.84 to 0.89 of each other, the tags are wrong more often than the models are, and where the
tags are wrong the models that read the ball tend to be wrong the same way.

Measurement only. Nothing here may be used to fit, tune or pick thresholds for a model.

```
MTAH_DATA_DIR=... python tagcheck/score.py sw=deep_sw_bar_ath.csv bt=deep_bt_bar_ath.csv ... --ref sw
```

## The checks

| check | what the tags say | what is counted | size |
|---|---|---|---|
| phase | `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION` name the team with the ball | share of the file's live frames inside the phase given to that team | 119 episodes, 40.9 min, 39% of graded frames |
| restart | a set piece (not a kick-off) follows a stoppage | a dead frame from 3 s before to 6 s after the tag opens; then, where the tag names a team, that team has the ball when play resumes | 53 tags, 27 with a team |
| switch | an attacking transition is the ball changing hands | a change to the tagged team from 3 s before to 8 s after the tag opens | 24 |
| kick-off | after a goal the ball is dead until the conceding team kicks off | dead share over the 15 s before the kick, team after it | 4 |

The `DEFENDING ...` and `DEFENSIVE TRANSITION` codes are the mirror of the attacking ones (same
intervals, other team) and add nothing. The place codes (`BALL IN FINAL THIRD`, `BALL IN THE BOX`,
`PLAYERS IN THE BOX`) do not name the team with the ball (44% to 54% agreement in the public
window) and are not used. Every throw-in is tagged `N/A`.

## How good are the tags

Against the public five minutes, the only place they can be checked:

- 8 tagged phases, 132 s of live play. The tagged team has the ball on **77%** of it. Seven phases
  agree 85% to 100%; one 33 s `PROGRESSION` is tagged for Athletic while Barcelona have the ball
  for 73% of it. So the true labels themselves would score 0.774 on the phase check there.
- 1 set piece and 1 attacking transition fall inside it; both agree. Nothing can be said about
  their accuracy from one case each.

## Scores, whole match

Phase share with a bootstrap over the 119 episodes; the difference is paired against `sw`.

| file | reads | phase team share | against sw | stoppage at tag (53) | tagged team restarts (27) | switch (24) |
|---|---|---|---|---|---|---|
| alldata_learned | ball | 0.890 [0.854, 0.923] | +0.022 [-0.001, +0.045] | 51 | 15 | 20 |
| deep v2 | players | 0.887 [0.851, 0.920] | +0.019 [+0.002, +0.040] | 49 | 21 | 22 |
| alldata_physics_tim | ball, Tim | 0.883 [0.849, 0.915] | +0.015 [-0.005, +0.038] | 51 | 21 | 22 |
| alldata_physics | ball | 0.879 [0.842, 0.912] | +0.011 [-0.010, +0.036] | 51 | 19 | 22 |
| alldata_physics_tags | ball, tags | 0.879 [0.842, 0.912] | +0.011 [-0.010, +0.036] | 51 | 19 | 22 |
| deep sw + tags | players, tags | 0.874 [0.830, 0.912] | +0.006 [-0.000, +0.019] | 50 | 22 | 22 |
| deep bt | players | 0.869 [0.826, 0.907] | +0.001 [-0.016, +0.017] | 48 | 22 | 22 |
| **deep sw** | players | 0.868 [0.824, 0.907] | reference | 48 | 20 | 22 |
| Tim's hmm v8 | ball | 0.862 [0.826, 0.896] | -0.006 [-0.033, +0.024] | 48 | 21 | 22 |
| deep sp | players | 0.857 [0.818, 0.891] | -0.011 [-0.027, +0.007] | 49 | 23 | 22 |
| hmm v7 + deep | ball | 0.841 [0.802, 0.876] | -0.028 [-0.058, +0.004] | 48 | 18 | 21 |
| baseline smooth125 | ball | 0.745 [0.696, 0.792] | -0.124 [-0.174, -0.074] | 27 | 8 | 19 |
| baseline raw | ball | 0.676 [0.643, 0.710] | -0.192 [-0.234, -0.150] | 41 | 12 | 22 |

Kick-offs after goals: every model file is dead for the whole 15 s before all four and gives the
conceding team the ball at three; at the second goal's kick-off the models are still dead 10 s
after the tag.

## Why this does not rank the candidates

1. **The intervals overlap.** Only v2 against sw has a paired interval clear of zero, and v2 is
   the older model that scores lower held out (0.524 against 0.558). Everything from 0.857 to 0.890
   is one group.
2. **The tags are noisier than the models.** In the public window the tags name the wrong team
   on 23% of live phase frames; the model files name the wrong team on 2% to 9% of the same frames.
   A referee that errs more often than the players cannot split them.
3. **The errors are shared.** In the public window the true team accuracy inside the phases and
   the phase share run opposite ways:

   | file | true team accuracy in the phases | phase share |
   |---|---|---|
   | deep sw | 0.974 | 0.714 |
   | alldata_physics | 0.967 | 0.774 |
   | alldata_learned | 0.934 | 0.779 |
   | Tim's hmm v8 | 0.907 | 0.794 |

   On the mis-tagged phase the players-only files give Athletic 27% (the truth is 27%), the files
   that read the ball 39% to 41%. The tags are made from the same ball track, so a file that
   follows the ball inherits some of their mistakes and is paid for it here. The phase check
   therefore leans towards ball-reading files by an amount that cannot be measured outside the
   five minutes.
4. **It does not see what separates the candidates.** The held-out differences come from short
   possessions and turnover timing. A phase is 17 s long on median and the check is a frame
   share; a file that drew no short possessions at all would lose nothing.
5. **Some files used the tags.** `deep sw + tags` and `alldata_physics_tags` read the set-piece
   tags, and Tim's stoppage settings were checked against them, so the restart columns are not an
   independent check for those.

## What it is good for

- **A whole-match smoke test.** The baselines land 0.12 to 0.19 below the models with intervals
  far from zero; a file with a broken half, swapped teams or a time shift would too. The public
  five minutes cannot see anything after minute five.
- **Where two files disagree.** Inside tagged phases sw and alldata_learned disagree on the team
  on 2,030 frames (3% of phase frames); the tag sides with alldata_learned on 66% of them. Given
  point 3 that is a lean, not a verdict.
- **Restart team.** alldata_learned gives the tagged team the restart at 15 of 27 set pieces
  against 19 to 23 for the others. Seven events against two the other way; by itself not
  significant, but it is the same weakness its public-window score showed.

Tim's file (`hmm_v8.csv`) was made with `python -m hmm.decode`, unchanged.
