# possession/research: what the classical literature adds

A survey of the non-deep-learning literature on reading possession from player tracking, and a
held-out test of each idea on the 30 outside matches. The learned and deep approaches are a
separate thread's. Everything here predicts from the players alone.

```bash
uv run python possession/research/feats.py     # 5 Hz feature cache, all 30 matches (~2 min)
uv run python possession/research/feats2.py    # lines, keeper, pressure
uv run python possession/research/feats3.py    # per-player features for the choice model
uv run python possession/research/lab.py       # one feature at a time, held out by provider
uv run python possession/research/expN.py      # the experiments below
```

`lab.py` is a fast stand-in for `external/heldout.py`: the same three folds and provider
weighting, features at 5 Hz, a logistic for who and one for dead, and the fixed minimum runs for
clean-up. It reproduces the two reference points (0.284 against 0.282 for main, 0.397 against
0.392 for the first fitted team model), so differences inside it carry over; anything adopted is
re-scored with `external/heldout.py`.

## What the literature says

| source | what it offers a players-only model |
|---|---|
| Frencken et al. 2011; Bartlett et al. 2012; Clemente et al. 2013 (centroid, stretch index, surface area) | The side with the ball is wider, longer and more spread; the other side compacts. Already the base of this model. |
| Kim, Grundmann, Essa et al., CVPR 2010 (motion fields) | The players' velocity field converges on where play is going. A stand-in for the ball. |
| Welch et al. 2021 and the collective-motion literature (polarisation) | A defending block moves as one; the side in possession does not. |
| Vidal-Codina et al. 2022 (rule-based event detection) | Set pieces have a shape: a player on the line, a packed box, everyone in his own half. |
| Rein et al. 2017; Fernandez & Bornn 2018 (Voronoi / pitch control) | Share of space each side controls. |
| Link & Hoernig 2017 (individual ball possession) | Needs the ball. Its idea of control by the closest unchallenged player became the pressure signal: defenders close on the man with the ball. |
| McFadden's conditional logit, as a hand-made version of Amirli & Alemdar 2022 | Score every player for being on the ball, softmax over the pitch. |
| Sarawagi & Cohen 2004 (semi-Markov models) | Decode whole spells with a length prior instead of a flat switch cost. |
| Kim et al. 2023 (Ball Radar); PathCRF 2026 | Deep players-only models reach 86% team accuracy (92% within one provider). The yardstick. |

## What held up, and what did not

Held-out mean spellF1 in the harness. The reference is width, length and push for who, and the
three speed signals for dead.

| idea | spellF1 | verdict |
|---|---|---|
| reference: width + length + push | 0.397 | |
| + mean speed difference | 0.408 | kept (already in the model by the time this was written) |
| + pressure on the most closed-down player | 0.405 | **kept**: the best new who signal |
| + polarisation | 0.399 | **kept** with pressure: together with speed +0.012, 25 of 30 matches up |
| + any one of 20 others (spread, centroid height, back line, front line, Voronoi share, nearest-opponent distances, free players, forward and backward runs, expansion rate, motion-field convergence, field position, keeper height and gap, back-four width and flatness, front-three width) | 0.396 to 0.401 | dropped: each inside ±0.004 |
| all 42 who features | 0.416 | dropped: +0.005 over six for 36 more |
| finer time windows (1, 2, 4, 8 s after and before) | 0.389 to 0.393 | dropped |
| odd interaction terms (signal x tempo, signal x field position, x\|x\|) | 0.414 to 0.418 | dropped |
| per-match self-training on confident frames | 0.407 to 0.414 against 0.431 | dropped: hurts IDSSE and Metrica |
| who-is-on-the-ball choice model over the players | 0.407 against 0.411 | dropped: picks the right player 45% of the time held out (chance is 5%) and his team 71 to 75%, which is below what shape gives |
| dead: + packed penalty area, nearest player to a corner flag | 0.418 | **kept** |
| dead: + summed spread, kickoff shape, 80th-percentile speed | 0.424 | kickoff shape **kept**; 30 of 30 matches up against the reference |
| dead: all 15 signals | 0.429 | dropped: breaks when players go missing (below) |
| dead: speed read 1.5 to 4 s ahead, to start stoppages earlier | 0.419 against 0.422 | dropped |
| Viterbi with switch costs picked held out | 0.428 against 0.429 | no difference from fixed minimum runs on these scores |
| semi-Markov decode with a length prior per label | 0.428 to 0.429 against 0.422 | not adopted: +0.006, the folds pick different settings, and the prior is wanted as weak as the grid allows |

## Three findings worth more than any one signal

**Team-aggregate signals stop at about 85% team accuracy.** Whatever was added, frame-level team
accuracy held out went from 0.843 (three signals) to 0.853 (all 42). Ball Radar, a deep model on
the same kind of input, reports 86%. The shape of two teams does not hold more than that.

**The team call is right when the possession is long and wrong around turnovers.** Team accuracy
by distance from the nearest true change of label (IDSSE / Metrica / SkillCorner):

| seconds from a change | under 1 | 1 to 2 | 2 to 3 | 3 to 5 | over 5 |
|---|---|---|---|---|---|
| team accuracy | 0.64 / 0.65 / 0.64 | 0.82 / 0.84 / 0.76 | 0.92 / 0.92 / 0.81 | 0.97 / 0.95 / 0.86 | 0.98+ / 0.93 / 0.90+ |

Possessions under 3 s are called for the wrong team more often than not (0.2 to 0.5). Half of
open-play turnovers have no predicted change of the right direction within 5 s, and those that do
are 2 s out (standard deviation). The providers also disagree on when a turnover happens: against
this model IDSSE's are 0.4 s early and SkillCorner's 0.4 s late. Shape takes seconds to answer a
turnover, and no window, interaction or decode tried here recovers that.

**Given the true stoppages the score would be 0.53; given the true team, 0.78.** So the stoppage
model is the smaller problem but still worth 0.1, and most of that is the edges of stoppages.

## Missing players

BAR-ATH tracks 10 or fewer of a side on a quarter of its frames. Players were knocked out of the
outside matches in runs (11% of player-frames) and the models scored on that:

| | complete | weights from complete, scored on knocked-out | fitted and scored on knocked-out |
|---|---|---|---|
| three who + three dead | 0.397 | 0.369 | 0.369 |
| six who + three dead | 0.411 | 0.382 | 0.388 |
| six who + eight dead | 0.422 | 0.389 | 0.397 |
| all 42 who + all 15 dead | 0.431 | 0.380 | 0.420 |

Every set loses about 0.03. The compact set keeps its lead; the full set loses it unless its
weights are fitted on tracking with players missing, because several of its dead signals count
players. Fitting the shipped weights on knocked-out tracking is the obvious next step for
BAR-ATH and has not been done.

## On the current model, with the official script

The harness gains were measured against the three-signal team model and fixed minimum runs. By
the time they were ported, `possession/model.py` had mean speed, players on a boundary line, both
teams' width and a Viterbi decode. On top of that, `external/heldout.py`:

| who signals | dead signals | spellF1 (IDSSE / Metrica / SkillCorner) | macroF1 | alive/dead | team |
|---|---|---|---|---|---|
| 4 (as shipped) | 5 (as shipped) | 0.435 (0.424 / 0.506 / 0.375) | 0.802 | 0.914 | 0.864 |
| + pressure, polarisation | 5 | 0.436 (0.423 / 0.507 / 0.379) | 0.804 | 0.915 | 0.866 |
| 4 | + packed box, corner flag, kickoff shape | 0.435 (0.423 / 0.506 / 0.375) | 0.803 | 0.914 | 0.864 |
| + pressure | + all three | 0.437 (0.425 / 0.510 / 0.376) | 0.804 | 0.914 | 0.866 |
| + pressure, polarisation | + all three | 0.436 (0.421 / 0.508 / 0.378) | 0.805 | 0.915 | 0.866 |

**None is adopted.** The most any combination adds is 0.002, against a bar of 0.01. The speed
difference and the boundary-line signal already carry what pressure and the set-piece shapes
carried in the harness. `possession/model.py` and `features.py` are left exactly as the model
thread has them; the signals stay here (`feats2.py`: `_pressure`, `feats.py`: `box_max`,
`corner_min`, `own_half`) for anyone who wants them as inputs to a learned model.

## What this says about where to look next

Players-only, linear, cross-provider, the score sits at 0.43 to 0.44 whichever classical route is
taken, and Tim's boosted trees on players-only features with a chain score about the same
(0.405 / 0.520 / 0.389). The ceiling is what team shape knows about a turnover in the two seconds
around it. Three things could move it, none of them a new aggregate signal:

1. The ball, when it is seen. Combining with `hmm/` is where the next large step is.
2. A model that reads individual trajectories around a turnover (the learned thread's space).
3. Fitting on tracking with players knocked out, so the weights suit BAR-ATH's gaps (above).

Sources: Vidal-Codina et al. 2022 (arXiv 2202.00804), Kim et al. 2023 (arXiv 2306.08206) and
PathCRF (arXiv 2602.12080) were read for this survey. The other papers in the table are cited
from memory and need checking before they go into the technical document.

## The two seconds around a turnover (follow-up)

**The players do answer a turnover fast, with their acceleration, not their shape.** Mean of each
signal around true open-play turnovers, signed so that positive follows home winning the ball,
in the signal's own standard deviations (`exp14.py`; IDSSE, the other two agree in shape):

| seconds after the turnover | -2 | -1 | 0 | +0.4 | +1 | +2 | +3 | +6 |
|---|---|---|---|---|---|---|---|---|
| everybody's acceleration along the pitch | -0.50 | -0.12 | +0.82 | +1.09 | +1.07 | +0.50 | +0.20 | -0.06 |
| players reversing direction | -0.31 | -0.11 | +0.40 | +0.75 | +0.87 | +0.24 | +0.07 | -0.05 |
| push (everybody's velocity) | -0.53 | -0.68 | -0.57 | -0.41 | -0.14 | +0.14 | +0.26 | +0.28 |
| width difference | -0.27 | -0.26 | -0.24 | -0.22 | -0.19 | -0.13 | -0.08 | +0.01 |

Acceleration peaks 0.4 to 1 s after the turnover at about one standard deviation on IDSSE and
Metrica (0.44 on SkillCorner, whose 10 fps broadcast tracking is smoother). Push only changes
sign 1.5 s after, and width takes 3 to 6 s. So the fast evidence is an impulse, and a model that
classifies each frame's level cannot use an impulse:

| idea | score | verdict |
|---|---|---|
| seven fast cues as team signals (mean, median and top-four acceleration, reversals, sprint onsets, brakes, turns over 90 degrees), each over 0.6 s and summed over 2 and 5 s after and before | harness 0.406 to 0.411 against 0.411; team accuracy within 1 s of a change 0.637 to 0.640 against 0.637 | dropped |
| the acceleration impulse as a cheaper change of team in the Viterbi (an input-dependent switch cost), harness | 0.411 against 0.402 for the same Viterbi without it: IDSSE +0.023, Metrica +0.014, SkillCorner -0.010 | tried on the model |
| the same on `possession/model.py`, `external/heldout.py turn_clue=0,1.5,3,6` (1.5 picked in all three folds) | **0.438** against 0.435: IDSSE 0.429 (+0.005), Metrica 0.514 (+0.008), SkillCorner 0.372 (-0.003) | not adopted: +0.003, one provider down |

The clue is the only thing in this survey that moved turnover timing at all, and it moved it on
the two optical 25 fps providers, which are the ones most like BAR-ATH's tracking. It is below
the bar on its own. The model-side code is in commit `0fb083f` (`turn_impulse` and the `turn` argument
of `viterbi`) and was taken back out of `possession/model.py`; it is the piece most worth
retrying inside a combined model, where the ball gives the decode something sharper to line the
impulse up with.

The knocked-out refit was stopped unfinished: the model thread had already run it (a refit
recovers 0.004; see `possession/README.md`).
