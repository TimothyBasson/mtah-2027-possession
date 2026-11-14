# deep/: a learned, players-only possession model

Reads the players' tracking only. The ball is never an input; it is a training target on the
outside matches. Separate from `possession/`, `hmm/` and `seq/`.

```bash
pip install -r requirements-deep.txt
python deep/train.py --fold idsse      # and metrica, skillcorner: one provider held out each
python deep/decode.py --tag v2         # THE score: held-out mean spellF1
python deep/train.py --fold none --steps 2000   # all 30 matches -> copy net.pt to deep/models/v2.pt
python deep/run.py                     # BAR-ATH -> evaluation/results/deep.csv
python deep/explain.py --match all     # each stage's output, for viz/combined/make_combined.py --deep
```

## v2 (2026-11-13): the pitch turned the right way, 0.529

`data.half_tensors` turned the pitch so home defends the left by comparing home's mean position
with halfway. A side that dominates a half averages beyond halfway, so 5 of the 60 outside halves
were turned the wrong way (4 SkillCorner, 1 IDSSE): the same bug `possession/features.own_goal`
had. On those halves v1's team accuracy was 0.70 against 0.82 on the rest of SkillCorner. Now the
two teams are compared with each other. BAR-ATH was read correctly either way.

Held out by provider, spellF1 (mean with macroF1 / alive-dead / team):

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| v1 | 0.541 | 0.635 | 0.381 | 0.519 / 0.839 / 0.935 / 0.880 |
| v1's nets, those halves predicted again (`predict.py --tag v1 --out v1o`) | 0.552 | 0.636 | 0.398 | 0.529 / 0.844 / 0.936 / 0.886 |
| **v2**: retrained with the fix, same seed | 0.554 | 0.633 | 0.398 | **0.529** / 0.845 / 0.936 / 0.886 |

The gain is the five halves being read correctly, not a better network: retraining on the
corrected halves changes nothing further. It is a fix to the same seed, so it is not subject to
the 0.013 seed-to-seed spread the variants below are. `models/v2.pt` is the all-30 model.

BAR-ATH public 5 minutes with `models/v2.pt`, nothing tuned on it (check only; 17 possessions,
so about ±0.2 of noise): spellF1 0.826, macroF1 0.918, 14 possessions against 17 (v1: 0.775,
0.894, 15). Full match: 281 possessions, 47% dead (v1: 280, 46%); the two agree on 97.6% of
graded frames.

**Regenerating the file.** `run.py` pins one thread. On the machine it was made on (Intel Xeon,
torch 2.15.0, Python 3.11) `python deep/run.py --threads 1`, `4` and `8` write byte-identical
CSVs (sha256 `5a94b7d8...0d813d`). It has not been checked on another CPU, where the last bits of
a probability could differ and a near-tie in the decode could move a boundary. `models/v2.pt`
came from one uninterrupted run of `train.py --fold none --steps 2000 --tag v2` (4 threads, seed
0); checkpoints now carry torch's generator as well, so a resumed run would match too. `v1.pt` is
no longer shipped: v2 replaces it, and `predict.py --tag v1` shows what the fix alone does.

**Against `hmm` v7 on BAR-ATH, no labels read** (`run.py`, `hmm/decode.py`, then
`compare.py --against evaluation/results/hmm_v7.csv`; asked for in the review of #29):

| | `deep` v2 | `hmm` v7 |
|---|---|---|
| half 1 | 130 possessions, median 7.4 s, 45% dead | 132, 8.1 s, 42% |
| half 2 | 151 possessions, median 6.3 s, 48% dead | 157, 6.2 s, 47% |
| public 5 minutes (check only, 17 possessions) | spellF1 0.826, macroF1 0.918, 14 possessions | 0.606, 0.783, 16 |

The two agree on 86.5% of graded frames: 84.5% where the ball is seen (59% of frames) and 89.4%
where it is missing. Where both call the ball live they name the same team on 89.4% as well (a
coincidence of two different counts: 0.8939 of 64,643 frames and 0.8936 of the both-live frames).
The disagreements are 5.3% of frames on the team (93% of those with the ball in view), 5.1% where
`deep` says dead and `hmm` live, and 3.0% the other way. So the two count about the same number
of possessions (281 and 289) and differ mostly on who has a ball that the tracker can see.

Those team disagreements are 183 short runs, 340 s in all: 50 under 1 s, 67 of 1 to 2 s, 58 of
2 to 5 s, 8 of 5 to 10 s (the longest 8.0 s, at 48:28 of the second half) and none longer. So
neither model has given a long possession to the wrong team where the other has it right; the
difference is brief possessions and where a change is placed, with the ball in view.

On the public window the bootstrap intervals overlap (`run_evaluation.py --bootstrap 1000`):
`deep` v2 spellF1 0.826 (0.63 to 0.95), `hmm` v7 0.606 (0.44 to 0.76). Nothing in `deep/` was
chosen on that window.

**Against `hmm` v7 held out, on the same 30 matches, paired by match** (`folded.py`). `hmm` v7
is `external/score_control.py`'s run at the shipped weight (its classifier trained without the
provider scored, its tables counted on the others, its hand-set weights as shipped, ball at 59%
seen); `deep` v2 is the fold net that never saw the provider, with the shipped decode in every
fold. "Folded" scores against the outside labels folded towards the challenge's
controlled-possession rule (`external/control.py`), which `hmm` v7's classifier is trained on and
`deep` is not.

| spellF1 | `deep` v2 | `hmm` v7 | `deep` − `hmm`, paired (standard error) | matches `deep` is ahead on |
|---|---|---|---|---|
| labels as they are: IDSSE (7) | 0.549 | 0.523 | +0.027 (0.010) | 6 of 7 |
| Metrica (3) | 0.635 | 0.769 | −0.134 (0.014) | 0 of 3 |
| SkillCorner (20) | 0.398 | 0.376 | +0.022 (0.006) | 17 of 20 |
| all 30 matches, each counting once | 0.457 | 0.450 | +0.008 (0.010) | 23 of 30 |
| mean of the three providers | 0.527 | 0.556 | −0.028 | |
| folded labels: IDSSE (7) | 0.675 | 0.609 | +0.066 (0.012) | 7 of 7 |
| Metrica (3) | 0.720 | 0.798 | −0.077 (0.004) | 0 of 3 |
| SkillCorner (20) | 0.451 | 0.428 | +0.023 (0.009) | 16 of 20 |
| all 30 matches, each counting once | 0.530 | 0.507 | +0.023 (0.010) | 23 of 30 |
| mean of the three providers | 0.615 | 0.611 | +0.004 | |

Which is ahead depends on how the matches are weighed, so neither average settles it. Match by
match `deep` is ahead on 23 of 30 under either labelling, and on every provider but one. That
one is Metrica, where `hmm` is ahead on all three matches by 0.08 to 0.13, and Metrica is the
set most like BAR-ATH: the same tracking system and the ball seen about 60% of the time. Three
matches are few, and their labels are rebuilt from on-ball events, which a ball-reading model
follows more closely than a players-only one can; but for the question of which file to ship
they are the most relevant three. Against that, on BAR-ATH's own public window `deep` is ahead
(above), on 17 possessions. The evidence does not pick a winner.

**Is the decode merging possessions?** (`compare.py --sweep`) The review noted 281 possessions
against about 350 if the whole match ran at the public window's density. Every setting, the same
in all three folds, with BAR-ATH's count beside its held-out score:

| switch cost | shortest run kept, H/A and D | IDSSE | Metrica | SkillCorner | mean | BAR-ATH possessions (median) |
|---|---|---|---|---|---|---|
| 0 | none | 0.537 | 0.608 | 0.355 | 0.500 | 403 (3.8 s) |
| 1 | none | 0.552 | 0.630 | 0.380 | 0.521 | 332 (5.3 s) |
| 1 | 0.5 s | 0.554 | 0.633 | 0.383 | 0.523 | 327 (5.4 s) |
| 2 | 0.5 s | 0.553 | 0.633 | 0.388 | 0.525 | 316 (6.0 s) |
| 2 | 1 s | 0.554 | 0.634 | 0.394 | 0.528 | 304 (6.3 s) |
| **4 (shipped)** | **1 s** | 0.549 | 0.635 | 0.398 | 0.527 | 286 (6.8 s) |
| 8 | 1 s | 0.544 | 0.630 | 0.405 | 0.526 | 269 (7.6 s) |
| 4 | 2 s and 3 s | 0.532 | 0.621 | 0.420 | 0.524 | 235 (8.6 s) |

(286 here is 281 once the frames before each kickoff are set to D.) From 235 to 332 possessions
the held-out mean stays within 0.521 to 0.528, and the providers pull opposite ways: IDSSE wants
a cheaper switch, SkillCorner a dearer one. That disagreement, not noise, is why the outside
matches cannot say how finely to cut (the settings are compared on the same probabilities, so
the differences are paired and small ones are real). Getting to 350 needs no decode at all and
costs 0.03. `hmm` v7 lands on 289 from the ball, by a different route, and 350 is five minutes
extrapolated to a match. So the shipped setting stays; cost 2 with 1 s runs (304 possessions,
0.528) is as good a choice, and nothing here should be picked on the public window.

**Players missing as on BAR-ATH** (`predict.py --knock`: `external/score.py drop_players`, about
7% of player-frames lost in runs and bursts, speeds recomputed):

| | full tracking | players knocked out |
|---|---|---|
| v1, direction fixed | 0.529 | 0.511 |
| v2 | 0.529 | 0.516 (IDSSE 0.547, Metrica 0.610, SkillCorner 0.392) |
| v3: v2 with half the training windows knocked out the same way (`train.py --knock 0.5`) | 0.524 | 0.516 (0.528, 0.624, 0.396) |

BAR-ATH's gaps cost about 0.013 to 0.018, the same as they cost `possession/` (0.016). Training
on knocked-out windows does not win it back (0.516 either way, and 0.005 lower on full
tracking), so v3 is not kept. The loss is information that is gone, not a network surprised by
gaps: v1 already hides up to 15% of players at random in training.

**With `possession/` in the same decode** (`stack/`): 0.530 against 0.529. The hand-built signals
add nothing the network has not learned, so this model stands alone as the players-only model.

## v1 (2026-11-12)

Held out by provider, the same folds and equal-provider mean as `external/heldout.py`:

| held out | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE (7) | 0.541 | 0.877 | 0.949 | 0.895 |
| Metrica (3) | 0.635 | 0.845 | 0.912 | 0.929 |
| SkillCorner (20) | 0.381 | 0.796 | 0.945 | 0.814 |
| **mean** | **0.519** | 0.839 | 0.935 | 0.880 |

For comparison, same protocol: `possession/` on main 0.282; the fitted team model with Viterbi on
`claude/project-thread-3lcb21` 0.435; the classical research branch 0.435 to 0.437.

Each fold trains on the other two providers minus a few validation matches (1 IDSSE, 1 Metrica,
3 SkillCorner where they are in training), and the decode settings are chosen on those validation
matches. The SkillCorner fold therefore learns from only 8 matches, which is the likely reason it
is the weakest; the model that labels BAR-ATH learns from all 30.

BAR-ATH, nothing tuned on it (check only; 17 possessions, so about ±0.2 of noise on spellF1):

| public 5 minutes | spellF1 | macroF1 | alive/dead | team acc | possessions |
|---|---|---|---|---|---|
| `deep` | 0.775 | 0.894 | 0.961 | 0.968 | 15 (truth 17) |
| `possession` (main) | 0.352 | 0.826 | 0.960 | 0.872 | 8 |

Full match: 130 / 150 possessions per half, median 7.8 / 6.6 s, 44% / 48% dead.

## What it is, and where it comes from

- **Players as an unordered set.** Each frame's players go through two self-attention blocks and
  are pooled per team (Set Transformer, Lee et al. 2019; used for exactly this in Kim et al., KDD
  2023, "Ball trajectory inference from multi-agent sports contexts using set transformer and
  hierarchical Bi-LSTM"). No player identity is needed, which suits our 46 swapping track slots.
  Per player: position, velocity, speed, change of speed, team.
- **Time.** Twelve dilated convolutions (0.2 s to 6.4 s steps, twice) and a two-way GRU over the
  5 fps sequence, as the offline task allows (MS-TCN, Farha & Gall 2019; TranSPORTmer, Capellera
  et al. 2024, which labels possession states jointly with trajectories).
- **The ball as a training target.** A second head predicts where the ball is, on live frames
  where it was seen. Kim et al. and TranSPORTmer both find the ball is largely recoverable from
  the players; here that knowledge is forced into the shared features and the head is thrown away.
  A third head marks frames within 0.4 s of any change of label.
- **Symmetry.** The pitch is turned so home defends the left; swapping the teams and mirroring
  the pitch are exact symmetries, used as augmentation and averaged over at prediction.
- **Robustness to tracking.** Position jitter (0.2 m) and hiding up to 15% of players per window.
- **Decode.** A three-state chain with one switching cost, then runs under 1 s absorbed. The
  score is flat across the decode grid (IDSSE 0.527 to 0.541, Metrica 0.630 to 0.636), so the
  network, not the decode, carries the result.

1,500 training steps per fold (2,000 for the final model) of 16 one-minute windows, CPU only,
about 70 minutes per fold on one core.

## What was tried on top (same folds, 1,500 steps, one seed each)

| variant | IDSSE | Metrica | SkillCorner | mean | kept? |
|---|---|---|---|---|---|
| **v1** (above) | 0.541 | 0.635 | 0.381 | **0.519** | yes |
| v1 again, another seed (`--seed 1`) | 0.520 | 0.618 | 0.380 | 0.506 | the same model: this is the run-to-run noise |
| no ball target (`--ball 0`) | 0.532 | 0.608 | 0.380 | 0.507 | no: inside the seed-to-seed spread, so the ball target's worth is not shown |
| v1 and no-ball probabilities averaged | 0.539 | 0.623 | 0.391 | 0.518 | no: nothing over v1 |
| short possessions weighted up in the loss (`--spell 0.5`) | 0.552 | 0.653 | 0.371 | 0.525 | no: +0.006, SkillCorner down 0.010 |

Two seeds of the same model differ by 0.013 in the mean (0.519 and 0.506), and up to 0.021 on a
fold. So none of the variants is distinguishable from v1, including the one without the ball
target, and the honest headline for this model is about 0.51, give or take 0.01. The gap to the
hand-built models (0.435) and the baseline (0.282) is many times that.

**Where v1 still loses.** Held out, share of true possessions matched at IoU 0.5, by length:

| | under 2 s | 2 to 5 s | 5 to 10 s | over 10 s | possessions predicted / true |
|---|---|---|---|---|---|
| IDSSE | 8% | 36% | 63% | 87% | 1,813 / 3,061 |
| Metrica | 10% | 44% | 67% | 81% | 714 / 1,041 |
| SkillCorner | 3% | 25% | 47% | 66% | 6,409 / 6,500 |

Long possessions are found; ones under 2 s almost never are, the same wall the hand-built model
hit. Weighting them up in the loss did not move it, so from players alone they may not be there
to find. BAR-ATH's public window has 2 of 17 possessions under 2 s, against 20 to 30% outside.

## Caveats

- The model that labels BAR-ATH is a single run (seed 0); its held-out twin scored 0.519, the other seed 0.506.
- The outside labels are not the organisers' rule (`external/README.md`); 0.519 is not a
  leaderboard prediction.
- `torch` is a new dependency, locked in `uv.lock` under the `deep` group (`uv sync --group deep`).
- A learned model can pick up a provider's tracking quirks. Holding a provider out guards
  against that, and the BAR-ATH public window agrees, but that window is five minutes.

## v4: trained on the folded labels (2026-11-14, not kept)

`deep/train.py --labels control --tag v4`, otherwise as v2 (same folds, steps and seed), scored by
`deep/folded.py --tag v4` with the shipped decode. The labels are folded towards the
controlled-possession rule (`external/control.py`), as `hmm` v7's classifier is trained.

| spellF1 | IDSSE | Metrica | SkillCorner | all 30 matches | provider mean |
|---|---|---|---|---|---|
| v2, labels as they are | 0.549 | 0.635 | 0.398 | 0.457 | 0.527 |
| v4, labels as they are | 0.526 | 0.612 | 0.421 | 0.464 | 0.519 |
| v2, folded labels | 0.675 | 0.720 | 0.451 | 0.530 | 0.615 |
| v4, folded labels | 0.666 | 0.717 | 0.488 | 0.553 | 0.624 |

SkillCorner gains 0.023 to 0.037 and the other two providers lose under both labellings, even on
the folded labels v4 was trained towards. One seed. A provider is down, so v2 stays. Against
`hmm` v7 nothing changes: v4 is still behind on all three Metrica matches (-0.157 raw, -0.081
folded).
