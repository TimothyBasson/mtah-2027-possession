# possession/: Jaedon's possession model

This model is separate from `hmm/`. The plan is to compare and combine the two at the end.

```bash
uv run python possession/run.py                    # -> evaluation/results/possession.csv (BAR-ATH)
uv run python external/heldout.py                  # THE score: per provider, fitted without it
uv run python external/score.py                    # every outside match, in-sample
uv run python possession/model.py --fit            # refit DEAD_FIT and TEAM_FIT on all 30 outside matches
uv run python viz/combined/make_combined.py --match all --explain --open   # see how it works
```

## How we judge a change

1. **Held out by provider.** `external/heldout.py` picks settings, and refits anything fitted
   (`model.fit`), on two providers, then scores the third, for each of IDSSE, Metrica and
   SkillCorner. The mean weights the three providers equally. A change counts only if it helps
   there: one that helps only on the provider it was tuned on is fitting that provider's labelling.
2. **Theory decides what goes in, data decides how much.** A signal needs a football reason
   before it is tried. Only a few weights are fitted, never per-match thresholds.
3. **Priors from football, not from our data**, such as possession counts and stoppage lengths.
4. **BAR-ATH is left alone.** The public 5 minutes are a final sanity check, never a target.
5. **Every idea goes in the log below**, with its held-out score, kept or dropped.
6. **A change counts from +0.01 on the held-out mean with no provider down.** Providers disagree
   by up to 0.025 on the same change.

## Experiment log (held-out mean over the three providers, `external/heldout.py`)

| date | idea | spellF1 | macroF1 | alive/dead | team | kept? |
|---|---|---|---|---|---|---|
| 2026-11-11 | v0 as on main: spread for who, median speed under 1.2 m/s for dead, short runs absorbed | 0.250 | 0.680 | 0.856 | 0.784 | baseline |
| 2026-11-11 | same, walking cutoff and minimum runs picked held out (1.4 m/s, 5 s, 1 s, the same in all folds) | 0.261 | 0.698 | 0.859 | 0.785 | not adopted: superseded by the next row |
| 2026-11-11 | **stoppage model**: logistic on 4 players-only speed signals (below) for dead | **0.266** | **0.712** | **0.879** | 0.790 | **yes** |
| 2026-11-11 | stoppage model, minimum runs re-picked held out | 0.268 | 0.715 | 0.883 | 0.790 | no: +0.002, folds disagree on the settings |
| 2026-11-11 | **+ acceleration**: 80th percentile of players' \|change of speed\|, ±3 s, as a fifth stoppage signal | **0.282** | **0.735** | **0.898** | 0.790 | **yes** |
| 2026-11-11 | a missing stoppage signal adds nothing instead of forcing dead (review of #22) | 0.282 | 0.735 | 0.898 | 0.790 | yes: same mean, the principled rule |
| 2026-11-12 | **three signals**: drop the 80th-percentile speed and the 10 s median (table below) | **0.282** | **0.736** | **0.898** | 0.790 | **yes**: as good with two fewer |
| 2026-11-12 | spread lean only: 1 s lean window and 1 s minimum possession (were 2 s, 2 s), from a football-prior grid ("Where the mistakes are" below) | 0.287 | 0.736 | 0.899 | 0.790 | yes on the spread lean; not carried over: the team model and Viterbi decode (above) replaced both settings |
| 2026-11-12 | **team model**: logistic on width, length, push and speed difference, each over the next 2 s and the last 2 s, instead of the spread's sign (below) | **0.392** | **0.775** | 0.898 | **0.854** | **yes** |
| 2026-11-12 | **Viterbi decode**: the best H/A/D path on the two probabilities, a change of team costing 5 and to or from dead 20, instead of cutting each at even odds and absorbing short runs | **0.404** | **0.780** | 0.899 | 0.856 | **yes** |
| 2026-11-12 | the same, the two costs picked held out from 2, 5 x 20, 40 | 0.405 | 0.782 | 0.901 | 0.856 | no: the folds pick three different pairs, the surface is flat |
| 2026-11-12 | **which goal a team defends**, read from the two teams against each other, not against halfway (a bug: wrong in a half of 9 of the 30 matches, which flipped push and dropped a forward as the keeper) | **0.414** | **0.783** | 0.899 | **0.861** | **yes** |
| 2026-11-12 | **two more stoppage signals**: players on or over a boundary line, and both teams' width summed, each ±1 s | **0.435** | **0.802** | **0.914** | 0.864 | **yes** |
| 2026-11-12 | restart taker as a team signal: home minus away players on a boundary line, over the last 2 / 5 / 10 s | 0.435 | | | | no: frame accuracy 0.846 → 0.847, width already says it |
| 2026-11-12 | decode variations: dead log-odds shifted by −1 / −0.5 / +0.5 / +1; team odds shifted 0.5 s earlier, 0.5 s or 1 s later; team evidence halved or doubled; costs 2-10 and 10-30 | 0.405-0.437, picked held out 0.432 | | | | no: nothing clears +0.01 and the folds disagree |
| 2026-11-12 | look-ahead stoppage signals (on a line, width, median speed, acceleration over the next 1-5 s, to start stoppages earlier), with and without players tracked | 0.436-0.439 | 0.804-0.810 | 0.914-0.918 | 0.865 | no: under +0.01, IDSSE falls 0.003-0.008 |
| 2026-11-12 | team model at more scales: push and width over 0.6 s or 2-5 s as well; goal-side marking; without length | 0.432-0.439 | 0.801-0.803 | 0.914 | 0.861-0.866 | no: under +0.01 |
| 2026-11-12 | start every decoded stoppage 0.5 / 1 / 1.5 / 2 / 3 s earlier (1.5 s picked in all three folds) | 0.440 | 0.807 | 0.912 | 0.869 | no: +0.005, up on all three but under +0.01; alive/dead falls |
| 2026-11-12 | players knocked out as BAR-ATH's tracker loses them (`heldout.py --knock score`), shipped weights: what the model loses to missing players | 0.419 | 0.792 | 0.912 | 0.851 | a measurement, not a change |
| 2026-11-12 | weights refitted on knocked-out tracking, scored on knocked-out tracking (`--knock both`) | 0.423 | 0.793 | 0.913 | 0.850 | no: recovers 0.004 of the 0.016, under +0.01; on full tracking that fit scores 0.436 / 0.802 / 0.915 / 0.862 (`--knock fit`) |

**Team model** (who has the ball, players only). The spread lean was one number with its cutoff
at zero, read over 2 s either side, so it answered a turnover about 2 s late. Now a logistic
model, fitted like the stoppage model and refitted per held-out fold, reads four signals
(`features.team_signals`, `TEAM_SIGNALS`), each home minus away with the keepers left out:

- **width** (sd of the players across the pitch): the side with the ball stretches the pitch, the
  side without it narrows to protect the middle
- **length** (the same along the pitch)
- **push**: both teams' mean velocity toward the goal home attacks. Everyone moves the way the
  ball is going
- **speed difference**: the side without the ball chases

Each is read twice, over the 2 s after the frame and the 2 s before, so eight weights and an
intercept. The after-window carries the weight and the before-window gets the opposite sign
(width +2.17 / −0.37, length +0.60 / −0.50, push +0.91 / −0.46): what the teams do next says who
has the ball now, and a team that is *becoming* wider has just won it. That is the turnover
signal, and it is what places turnovers on time. Speed difference is −0.32 / −0.18. The intercept
(+0.16) is home advantage: home has the ball 51-53% of live frames on the outside matches.

Per held-out provider, spread lean → team model:

| held out | spellF1 | macroF1 | alive/dead | team |
|---|---|---|---|---|
| IDSSE | 0.237 → 0.371 | 0.757 → 0.810 | 0.914 | 0.769 → 0.852 |
| Metrica | 0.309 → 0.460 | 0.729 → 0.779 | 0.875 | 0.804 → 0.889 |
| SkillCorner | 0.300 → 0.345 | 0.721 → 0.736 | 0.905 | 0.798 → 0.821 |

How the signals were chosen. Frame by frame on live frames, held out by provider (accuracy of the
sign, mean of the three), each signal alone over 2 s either side: width 0.805, spread 0.784,
length 0.714, goal-side marking 0.696, speed difference 0.654, each team's push 0.60, territory
0.559, relative centroid 0.517, back-line height 0.507. Width alone, after and before: 0.822.
Then end to end (held-out spellF1 / team, on a research copy of the pipeline, 3 s windows):

| signals (each after and before) | spellF1 | team |
|---|---|---|
| width | 0.354 | 0.834 |
| width, length | 0.354 | 0.833 |
| width, push | 0.385 | 0.847 |
| width, length, push | 0.389 | 0.850 |
| spread, push | 0.381 | 0.841 |
| width, length, push, territory | 0.390 | 0.850 |
| width, length, push, goal-side marking | 0.388 | 0.849 |
| **width, length, push, speed difference (kept)** | **0.400** | **0.856** |
| the same with each team's push separately | 0.401 | 0.856 |
| those four, territory and back-line height | 0.403 | 0.857 |

Windows of 1 / 1.5 / 2 / 2.5 / 3 s scored 0.404 / 0.408 / 0.407 / 0.405 / 0.400, so the existing
2 s stays. Taking each half's median out of a shape signal (to remove formation) lost 0.010
frame accuracy and was dropped. As with the stoppage model, which signals and the window were
chosen on all 30 matches; only the weights are refitted held out. SkillCorner gains least: its
10 fps broadcast tracking extrapolates off-camera players, which blurs push.

On BAR-ATH (check only): 138 / 141 possessions per half (was 99 / 113; the public window's
density suggests about 360 in all), mean 14.3 / 13.1 s, 38% / 42% dead, home 65% / 60% of live
frames; public 5 minutes macroF1 0.826 → 0.858, spellF1 0.352 → 0.538, 15 possessions against 17.

**Viterbi decode** (`model.viterbi`). Until now each model was cut at even odds and runs under
2 s / 3 s were absorbed. Now the labels are the single best H/A/D path: each frame scores the log
of its label's probability from the two models, a change of team costs 5 and a change to or from
dead costs 20. A lone frame can't flip the label, a run of evidence can, and the change lands
where the evidence turns. The costs are in the same units as the models' log-odds summed over
frames, and frames 0.04 s apart share most of their evidence, so they are not transition
probabilities. Per held-out provider (spellF1): IDSSE 0.371 → 0.385, Metrica 0.460 → 0.475,
SkillCorner 0.345 → 0.354. On a research copy of the pipeline, held-out spellF1 over team cost 2 /
3.5 / 5 / 8 and dead cost 10 / 20 / 40 / 80 stayed between 0.401 and 0.410: flat, so 5 and 20
were fixed rather than tuned. `decode=absorb` brings back the old clean-up.
On BAR-ATH (check only): 129 / 141 possessions per half, 39% / 43% dead; public 5 minutes macroF1
0.860, spellF1 0.538, 15 possessions against 17.

**Which goal a team defends** (`features.own_goal`) was read from which side of halfway its
players stood on average over the half. A side that dominates a half averages beyond halfway, so
both teams were given the same goal in a half of 9 of the 30 outside matches. There push pointed
the wrong way and the "keeper" left out was the furthest forward. Now the team standing nearer
x=0 defends that end. BAR-ATH was read correctly either way. Per held-out provider (spellF1,
team): IDSSE 0.385 → 0.402, 0.853 → 0.860; Metrica 0.475 → 0.484, 0.892 → 0.899; SkillCorner
0.354 → 0.356, 0.822 → 0.824. Refitted, push carries more (+1.20 / −0.57) and length less
(+0.24 / +0.02). The signal tables above were made before this fix.

**Two more stoppage signals.** Swapping in the true dead frames was worth about 0.10 spellF1
(below), so the stoppage model was revisited after the decode. Thirteen players-only signals were
each added to the three speed signals and scored frame by frame, held out (accuracy 0.864, dead F1
0.792 before):

| added, ±1 s | accuracy | dead F1 |
|---|---|---|
| **players within 0.5 m of a touchline or goal line, or beyond it** | 0.874 | 0.808 |
| **both teams' width (sd across the pitch), summed** | 0.874 | 0.807 |
| players inside either penalty area | 0.871 | 0.803 |
| players tracked | 0.866 | 0.794 |
| all players' mean velocity, or how aligned their velocities are | 0.865 | 0.793 |
| nearest player to a touchline; fastest player; 20th-percentile speed; share standing still; mean distance to the nearest opponent; both teams' length | 0.864-0.865 | 0.792-0.793 |
| change in the number tracked | 0.858 | 0.777 |
| **on a line + width (kept)** | **0.887** | **0.826** |
| on a line + width + penalty areas | 0.887 | 0.827 |

A player on a line is taking a throw-in or a corner, or fetching the ball: it pushes toward dead
(+0.78). Summed width pushes toward live (−0.78): in play both sides spread to use and to cover
the pitch, and at a stoppage they close up around the restart or walk back into shape. Per
held-out provider (spellF1, alive/dead): IDSSE 0.402 → 0.424, 0.921 → 0.932; Metrica 0.484 →
0.506, 0.870 → 0.889; SkillCorner 0.356 → 0.375, 0.906 → 0.922.

Do they stand on the same footing on BAR-ATH? Checked, no tuning. Share of frames with a player
on a line (all / alive / dead) and median summed width in metres (all / alive / dead): IDSSE
0.28 / 0.12 / 0.48 and 23.5 / 24.9 / 21.9; Metrica 0.24 / 0.15 / 0.42 and 25.4 / 26.1 / 23.7;
SkillCorner 0.21 / 0.12 / 0.43 and 23.7 / 24.6 / 21.5; BAR-ATH whole match 0.23 and 24.2, and in
its public minutes 0.19 alive against 0.35 dead, 28.6 alive against 21.3 dead. Same levels, same
direction. BAR-ATH (check only): 121 / 137 possessions per half, 41% / 42% dead; public 5 minutes
macroF1 0.846, spellF1 0.516, 14 possessions against 17 (0.860 and 0.538 before; one possession
of 17 is worth more than that gap).

**Missing players.** BAR-ATH is the match that is scored, and its tracker loses players; IDSSE
and Metrica never do. Measured on BAR-ATH's tracking alone, no labels: 22 are tracked on 50% of
graded frames, 21 on 26%, 20 or fewer on 23%; 18 times an hour most of them go at once (14 or
fewer tracked, 8.6 on average, median 2.7 s, mean 10.4 s), 5% of frames; outside those bursts
0.85 of the 22 are missing on average, in runs of median 3 s. That is 7% of player-frames.
`external/score.py drop_players` reproduces it on the outside matches (6.8% of player-frames,
two or more missing on 23% of frames, 14 or fewer tracked on 5%), and `heldout.py --knock` uses
it when scoring, fitting or both. Its shape is `possession/research/exp9.py`'s on
`claude/project-thread-uvbxom`, with BAR-ATH's rates in place of that script's 11%.

| weights fitted on | scored on | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|---|
| full tracking (shipped) | full | 0.424 | 0.506 | 0.375 | 0.435 |
| full tracking (shipped) | knocked out | 0.403 | 0.493 | 0.361 | 0.419 |
| knocked out | knocked out | 0.407 | 0.501 | 0.362 | 0.423 |
| knocked out | full | 0.426 | 0.509 | 0.372 | 0.436 |

So **0.435 is on full tracking; with BAR-ATH's gaps the same model scores 0.419** (macroF1
0.792, alive/dead 0.912, team 0.851). The loss is on the team side (0.864 → 0.851); the
stoppage model barely moves. Refitting on knocked-out tracking recovers 0.004, under the +0.01
bar, so the shipped weights stay. SkillCorner's off-camera players are extrapolated, so its
"full" tracking is already kinder than a real feed.

**Where the rest of the error is.** With the Viterbi labels, swapping in the true team wherever
both are alive gives held-out spellF1 0.80-0.86; swapping in the true dead frames gives 0.47-0.58.
So the team side is still most of the gap. By the true possession's length, the team call is
right on 96% / 90% / 90% of frames (IDSSE / Metrica / SkillCorner) in possessions over 16 s, about
75% in those of 4-8 s, and no better than a coin toss in those under 4 s, which are 14% of
IDSSE's live frames and 7% of the others'. A possession of a touch or two never changes the
teams' shape. Without the ball those are out of reach, and they are half of IDSSE's possessions.
No other team-level signal added to the four kept (frame accuracy 0.842 held out): box
containment, range instead of sd, back-line and front-line width, distance to the nearest
opponent, free men, closing speed, which centroid leads the other. Gradient boosting on all of
them over four windows reached 0.850, so the four are close to what team-level shape can say.

Stoppage model, per held-out provider (spellF1, alive/dead): IDSSE 0.199 → 0.217, 0.844 → 0.883;
Metrica 0.271 → 0.299, 0.849 → 0.865; SkillCorner 0.280 → 0.284, 0.874 → 0.889. Frame by frame
before any clean-up, held out, dead/live accuracy went 0.789 → 0.839 and dead F1 0.687 → 0.761.
Nine signals were tried; each one left out of the nine cost at most 0.006 accuracy, so four with
a clear reason were kept (0.839 against 0.841 for all nine). Tim's `hmm/clues.py` gave the
12 m/s cap on a player's speed; its ball-based dead clues stay out of this model.

**Acceleration** (Jaedon's idea: the fastest 20% carried the most weight, so look at how they
change pace too). In play players keep changing pace as they react to the ball; at a stoppage
they jog or walk steadily. Held out, frame by frame, added to the four speed signals:

| added | accuracy | dead F1 |
|---|---|---|
| nothing | 0.835 | 0.752 |
| change in the 80th-percentile speed, 2 s after minus before | 0.836 | 0.754 |
| fastest 20% of players: signed acceleration | 0.835 | 0.752 |
| fastest 20% of players: \|acceleration\| | 0.855 | 0.780 |
| 80th percentile \|acceleration\|, ±0.5 s | 0.861 | 0.789 |
| **80th percentile \|acceleration\|, ±3 s (kept)** | **0.864** | **0.793** |
| all ten | 0.866 | 0.794 |

How hard players change pace matters, not which way. Held-out spellF1 per provider: IDSSE
0.217 → 0.234, Metrica 0.299 → 0.311, SkillCorner 0.284 → 0.299. With it, all five weights point
the same way (faster or livelier means in play), and acceleration carries the most (−1.35).
On BAR-ATH (check only): 101 / 111 possessions per half, 38% / 42% dead (was 43% / 44%); public 5
minutes macroF1 0.814 → 0.820, spellF1 0.336 → 0.352, still 8 possessions against 17, 15% dead
against 20% true.

**Is it measuring the tracker's jitter?** Acceleration is a second difference of positions, so
positional noise raises it, and a noisier tracker would look livelier. Checked, no tuning. The
signal (m/s², 10th / 50th / 90th percentile) has the same floor on BAR-ATH as on the outside
data, and in the public minutes it separates dead from alive as it does there:

| | all frames | alive, median | dead, median |
|---|---|---|---|
| IDSSE | 0.40 / 0.91 / 1.48 | 1.19 | 0.48 |
| Metrica | 0.39 / 0.90 / 1.33 | 1.04 | 0.53 |
| SkillCorner | 0.53 / 1.00 / 1.43 | 1.10 | 0.66 |
| BAR-ATH, whole match | 0.43 / 0.99 / 1.61 | | |
| BAR-ATH, public minutes | 0.71 / 1.21 / 1.68 | 1.32 | 0.71 |

Taking each match's own floor out (subtracting its 10th or 50th percentile) scores the same held
out (0.282, 0.283), so the signal stays as it is. No graded BAR-ATH frame is missing any signal.

**Which signals still earn their place, once acceleration is in?** Each left out in turn, and
some pairs, refitted and scored held out:

| left out | spellF1 | macroF1 | alive/dead |
|---|---|---|---|
| nothing (five signals) | 0.282 | 0.735 | 0.898 |
| median speed | 0.281 | 0.734 | 0.897 |
| 80th-percentile speed | 0.280 | 0.735 | 0.898 |
| median, ±10 s | 0.282 | 0.736 | 0.899 |
| speed change | 0.278 | 0.732 | 0.896 |
| acceleration | 0.266 | 0.711 | 0.879 |
| median + 80th-percentile speed | 0.271 | 0.724 | 0.891 |
| **80th-percentile speed + median ±10 s (kept)** | **0.282** | **0.736** | **0.898** |
| 80th-percentile speed + speed change | 0.279 | 0.732 | 0.896 |

Acceleration took over the 80th-percentile speed's job (picking out the busiest players), and
the 10 s median added nothing on top. One of the two speed levels is needed, not both. So three
signals: median speed, speed change, acceleration. Per provider: IDSSE 0.237, Metrica 0.309,
SkillCorner 0.300. On BAR-ATH (check only): 99 / 113 possessions per half, 38% / 42% dead; public
5 minutes macroF1 0.826, spellF1 0.352, 8 possessions against 17.

**Not every choice was made held out.** The weights are refitted per held-out provider, but which
signals to use (4 of 9 speed signals, then 1 of 6 acceleration signals, then dropping 2 of the 5)
and their window lengths were chosen on all 30 matches. The ablations say those choices move held-out accuracy by under
0.01; the technical document should still say so.

Before the acceleration signal, on BAR-ATH, as a check only: 101 / 108 possessions per half, mean 18.1 / 16.5 s, 43% / 44% dead;
public 5 minutes macroF1 0.814, spellF1 0.336, 8 possessions against 17. Fewer possessions and
more dead than before: the model now merges more, which the next work on turnovers should balance.

## Where the mistakes were, on the spread lean (2026-11-12)

Measured on the model before the team model and the Viterbi decode, which were built in parallel
and replaced both settings below; `external/errors.py` runs on whatever the model now is.

On the model as it now stands (team model, Viterbi decode, five stoppage signals), the same
breakdown: wrong frames 25.4% → 19.1%; wrong team near a turnover 7.1% → 5.5%, mid-possession
5.1% → 2.6%; stoppage called in open play 4.9% → 4.2% (Metrica still 9.2%); true possessions
matched 20% → 36%, merged 35% → 26%, none of that team's frames 34% → 29%; predicted per true
possession 0.59 → 0.70; true turnovers found within 10 s 38% → 55%, now on time (median switch
−0.2 to +0.5 s). What is left is mostly the brief possessions and Metrica's open-play stoppages.

`uv run python external/errors.py` predicts each provider held out and sorts every wrong graded
frame, and every true possession it fails to match at IoU 0.5. With the 2 s settings (columns:
mean, IDSSE, Metrica, SkillCorner):

| graded frames wrong | 25.4% | 22.1% | 26.7% | 27.4% |
|---|---|---|---|---|
| wrong team, within 2 s of a true turnover | 7.1% | 8.0% | 6.6% | 6.9% |
| wrong team, mid-possession | 5.1% | 4.2% | 4.4% | 6.6% |
| stoppage called in open play | 4.9% | 1.3% | 10.3% | 3.2% |
| stoppage called, within 2 s of a true edge | 0.9% | 0.3% | 1.6% | 0.8% |
| missed stoppage: start / middle / end | 7.4% | 8.3% | 3.8% | 10.0% |

| true possessions (IoU 0.5) | mean | IDSSE | Metrica | SkillCorner |
|---|---|---|---|---|
| matched | 20.4% | 16.5% | 23.4% | 21.1% |
| merged into a prediction over twice as long | 34.5% | 40.2% | 29.2% | 34.1% |
| no prediction for that team overlaps it at all | 34.3% | 36.0% | 33.1% | 33.8% |
| split / shifted | 10.9% | 7.3% | 14.3% | 11.0% |
| predicted / true possessions | 0.59 | 0.46 | 0.56 | 0.75 |

- Half the errors are the team, half the stoppage. The team errors cluster at turnovers: the
  model finds 38% of true changes of team within 10 s, and those it finds come 0.6-2 s late.
- A third of true possessions get no frame of their own team at all. True possessions are short
  (median 3.7 / 5.9 / 5.8 s; 20-30% under 2 s, 38-53% under 4 s), and a team's shape doesn't
  change in a one-second possession, so spread can't see these whatever the smoothing.
- Metrica's "stoppage in open play" (10%) is the largest single bucket on any provider.

**Is the smoothing causing the merging?** The lean window and the minimum runs were set by hand
and never scored held out. Grid from football, not from the data: lean window 0.5 / 1 / 2 / 3 s
either side (a team reshapes within 1-3 s of a turnover), minimum possession 0 / 0.5 / 1 / 2 / 3 s
(a controlled touch and pass takes about a second), minimum stoppage 1 / 3 / 5 s (the quickest
restart, a quick throw-in, takes a few seconds). Held-out spellF1, each provider in its own fold,
providers equal (minimum stoppage 3 s; 1 s is 0.008-0.012 worse everywhere, 5 s the same as 3 s):

| lean window \ min possession | 0 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 0.5 s | 0.286 | 0.287 | 0.287 | 0.285 | 0.282 |
| 1 s | 0.286 | 0.287 | **0.287** | 0.284 | 0.281 |
| 2 s (was) | 0.283 | 0.283 | 0.283 | **0.282** | 0.280 |
| 3 s | 0.280 | 0.280 | 0.280 | 0.279 | 0.279 |

macroF1 is 0.735-0.737 across the whole grid. Picked on the other two providers, the folds chose
(1, 1, 3), (0.5, 1, 5) and (0.5, 0.5, 3): different points, but all on the same short plateau.
The middle of it, 1 s and 1 s, raises every provider (IDSSE 0.237 → 0.243, Metrica 0.309 →
0.317, SkillCorner 0.300 → 0.301) and was adopted. The minimum alive run is the same setting, so
it is 1 s too.

**So no: the smoothing isn't the cause.** The whole grid spans 0.279-0.287, possessions predicted
per true one go only 0.59 → 0.62, merged 34.5% → 33.6%, and turnovers found within 10 s 38% →
41%. The ceiling on these knobs is about +0.005; the merging comes from the spread signal not
seeing turnovers, which is where the next work goes (faster turnover signals). On BAR-ATH (check
only): 223 possessions of mean 17.2 s (was 212 of 18.0 s); public 5 minutes macroF1 0.824 (was
0.826), spellF1 0.352 (same), 8 possessions against 17.

## The model: players only, no ball

The model never reads the ball. Tim's `hmm/` works from the ball, so the two see the match through
different eyes and should make different mistakes, which is what makes combining them worth it.
`model.explain(F)` returns `p_home` (P(home has it | alive)) and `p_dead` per frame beside the
labels; those two are what a combination would take.

| held out (`external/heldout.py`) | spellF1 | macroF1 | alive/dead | team |
|---|---|---|---|---|
| IDSSE (7) | 0.424 | 0.837 | 0.932 | 0.864 |
| Metrica (3) | 0.506 | 0.800 | 0.889 | 0.899 |
| SkillCorner (20) | 0.375 | 0.770 | 0.922 | 0.829 |
| mean of the three | 0.435 | 0.802 | 0.914 | 0.864 |

On each frame:
- **Who** (`TEAM_SIGNALS`, `TEAM_FIT`): a logistic model on four signals, home minus away with the
  keepers left out, each over the next 2 s and the last 2 s: width, length, push (both teams'
  velocity toward the goal home attacks) and mean speed. The team with the ball stretches the
  pitch, everyone moves the way the ball is going, and the side without it chases. `team=spread`
  brings back the old rule (the more spread-out team, 2 s either side).
- **Dead** (`DEAD_SIGNALS`, `DEAD_FIT`): a logistic model on five signals:
  - median player speed over 0.5 s either side: the typical player walks at a stoppage
  - speed over the next 2 s minus the last 2 s: everyone slows into a stoppage and speeds up out
  - the 80th percentile of players' |change of speed| over 3 s either side: in play players keep
    changing pace as they react to the ball; at a stoppage they jog or walk steadily
  - players on or over a boundary line, 1 s either side: a throw-in or corner taker, or someone
    fetching the ball
  - both teams' width, summed, 1 s either side: in play both sides spread; at a stoppage they
    close up
  `dead=median` brings back the first rule (median speed under 1.2 m/s).
- Both models' weights are fitted on the 30 outside matches, each provider counted equally
  (`uv run python possession/model.py --fit`), and refitted without the held-out provider for
  every score above. A missing signal adds nothing either way.
- **Labels** (`model.viterbi`): the best H/A/D path on the two probabilities, a change of team
  costing 5 and to or from dead 20, within each half. With no team evidence (a side untracked)
  possession stays with the team that had it. `decode=absorb` brings back the old clean-up
  (even-odds cuts, runs under 2 s / 3 s absorbed).
- Graded frames before each kickoff are D on BAR-ATH.

Windows look forward as well as back (the task is offline) and stop at the edge of each half.

What it gets wrong:
- **Brief possessions.** A possession of a touch or two (under about 4 s) never changes the
  teams' shape, and the team call there is a coin toss. They are half of IDSSE's possessions.
  Each one missed also merges its two neighbours. This is the largest part of the remaining gap
  and likely needs the ball.
- **Turnover timing.** On long possessions the call is 69% right in the first half-second and
  over 90% after 2 s on IDSSE and Metrica: shape still answers a turnover late.
- **Stoppages start late.** The players slow down after the whistle, not before it. Look-ahead
  signals did not fix it (log above).
- **SkillCorner.** Its off-camera players are extrapolated, which blurs push; it gains least.
- **Signal choice was made on all 30 matches**; only the weights are refitted held out.

## v0, as first built (kept for the record; the numbers below are from 2026-11-05 to 11-11)

The first version: the more spread-out team has the ball (spread is each team's mean distance from
its own centre over 2 s either side, keepers left out), the frame is dead when the median player
speed is under 1.2 m/s, and alive runs under 2 s, dead runs under 3 s and team runs under 2 s
are absorbed (`absorb_short`), alive/dead and the team lean cleaned separately so a dead blip
can't move a turnover.

| outside matches | spellF1 | macroF1 | alive/dead | team |
|---|---|---|---|---|
| IDSSE (7) | 0.199 | 0.677 | 0.844 | 0.761 |
| Metrica (3) | 0.271 | 0.679 | 0.849 | 0.794 |
| all 10 | 0.221 | 0.677 | 0.846 | 0.771 |
| all 10, before absorbing short runs | 0.194 | 0.665 | 0.830 | 0.771 |

spellF1 per match, before and after absorbing:

| match | before | after |
|---|---|---|
| idsse_J03WMX | 0.192 | 0.188 |
| idsse_J03WN1 | 0.150 | 0.142 |
| idsse_J03WOH | 0.204 | 0.203 |
| idsse_J03WOY | 0.186 | 0.175 |
| idsse_J03WPY | 0.184 | 0.210 |
| idsse_J03WQQ | 0.212 | 0.258 |
| idsse_J03WR9 | 0.180 | 0.218 |
| metrica_1 | 0.208 | 0.233 |
| metrica_2 | 0.201 | 0.255 |
| metrica_3 | 0.227 | 0.325 |

It rose on 6 of the 10. The 4 that fell are IDSSE, by at most 0.011; DFL's spells are short
(`external/README.md`). The +0.027 mean is inside the ~0.04 single-match band, and most of it is
Metrica. The cutoffs were picked on the 10 outside matches: spellF1 is 0.217-0.222 for alive and
team runs of 1-2 s and dead runs of 3-5 s. An earlier version absorbed runs of the combined H/A/D
label into the longer neighbour; it scored the same (0.222) but could move a turnover by a few
seconds when a dead blip sat next to it.
Hysteresis was tried instead and did not help: a dead band on the team lean (0.25-1 m) or on the
walking cutoff (symmetric or one-sided, 0.1-0.6 m/s) scored 0.142-0.205 alone and no better than
0.217 on top of absorbing.

With the keepers counted (`external/score.py keepers=1`) it was 0.182 / 0.655 / 0.830 / 0.752.
Leaving them out raised team accuracy on 9 of the 10 matches.

For comparison, the earlier nearest-player-to-ball v0 scored spellF1 0.057 and macroF1 0.532 on all
10 (`--seen 0.59`). `--seen` does nothing here, because the ball is never read.

Possessions (H/A spells as the metric counts them), predicted against true:

| match | true | before | now |
|---|---|---|---|
| idsse_J03WMX | 511 | 378 | 215 |
| metrica_1 | 351 | 408 | 230 |
| metrica_3 | 375 | 434 | 224 |

So on the outside matches it now makes about half as many possessions as the truth, yet scores
higher. Absorbing helps Metrica (over-split before) far more than IDSSE (already under). The
missing possessions are mostly missed turnovers in the lean, not real short possessions absorbed:
only 2 of the 21 runs in the public window's truth are under the cutoffs.

On BAR-ATH, nothing tuned on it:

| | value |
|---|---|
| public 5 minutes (`run_evaluation.py --bootstrap 1000`), a pipeline check only | macroF1 0.772 (best on the board), spellF1 0.372 (5-95% 0.19-0.57), 12 possessions of mean 20.2 s against 17 of 14.2 s, 19.1% D against 19.6%; before absorbing: 0.724, 0.293, 24 of 9.9 s |
| full match, first / second half | 124 / 131 possessions, mean 16.5 / 14.8 s, 36% / 39% dead; before absorbing: 271 / 249, 7.2 / 7.5 s |
| graded frames with no spread lean (a side untracked for the whole window) | 1,277 (0.8%); they keep the previous team |
| graded frames with no player speed (counted as dead) | 0 |

The public window has only 17 possessions, so gaps on its spellF1 under about 0.39 are noise.
At the public window's density (17 in 5 minutes, about 106 graded minutes) a full match has
roughly 360 possessions, so 255 is about 30% too few, where 520 was about 45% too many.

This model is one half of a combined model with `hmm/`, not a replacement for the best single
model.

## Ideas to try (research, 2026-11-05)

### Ideas from the HMM approach
- **Touch check:** a nearby player only counts if the ball changes speed or direction (databallpy).
- **Event tags as anchors:** BAR-ATH only.
- **Gradient-boosted model for the clues:** trained on the outside matches.

### Different ways to frame the problem
1. **Choose spells, not frames.** Decode with a segment-level dynamic program: a learned spell-length prior plus a per-spell penalty, instead of a fixed switch cost. That length prior is what Viterbi's flat cost can't express. It could also pick the labels that maximise expected spellF1. It is cheap, and every other idea plugs into it. Background: semi-Markov CRF, Sarawagi & Cohen 2004.
2. **Spot turnovers, then fill the gaps.** Detect the moments possession changes. Useful cues:
   - both team centroids reverse direction
   - one team spreads out while the other compacts
   - the nearest players swap teams

   Then give each stretch between turnovers one label. Change-point detection (`ruptures`/PELT) is a label-free first version. Background: Vidal-Codina et al. 2022, the Kaggle DFL Shootout and SoccerNet action spotting.
3. **Team shape says who is attacking, without the ball.** The team in possession stretches, and the defending team compacts and presses. About 20 shape features per team cover the 41% of frames where the ball is missing.
4. **Rebuild the ball through gaps.** A switching filter with a "carried by player i" mode and a "free flight" mode, smoothed both ways, so where the ball reappears says who had it.
5. **Spell proposals plus a scorer** that sees a whole candidate spell, in the style of ActionFormer/BMN.
6. **Dead ball from the video:**
   - camera cuts and replays on the broadcast
   - the share of green pixels
   - optical flow

   There are no outside matches with video to validate this on.
7. **Pitch control around where the defenders converge,** as a stand-in ball position.

Top three for spellF1: 1, 2 and 3.
