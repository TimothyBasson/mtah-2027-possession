# possession/: Jaedon's possession model

This model is separate from `hmm/`. The plan is to compare and combine the two at the end.

```bash
uv run python possession/run.py                    # -> evaluation/results/possession.csv (BAR-ATH)
uv run python external/heldout.py                  # THE score: per provider, fitted without it
uv run python external/score.py                    # every outside match, in-sample
uv run python possession/model.py --fit            # refit DEAD_FIT on all 30 outside matches
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

## Experiment log (held-out mean over the three providers, `external/heldout.py`)

| date | idea | spellF1 | macroF1 | alive/dead | team | kept? |
|---|---|---|---|---|---|---|
| 2026-11-11 | v0 as on main: spread for who, median speed under 1.2 m/s for dead, short runs absorbed | 0.250 | 0.680 | 0.856 | 0.784 | baseline |
| 2026-11-11 | same, walking cutoff and minimum runs picked held out (1.4 m/s, 5 s, 1 s, the same in all folds) | 0.261 | 0.698 | 0.859 | 0.785 | not adopted: superseded by the next row |
| 2026-11-11 | **stoppage model**: logistic on 4 players-only speed signals (below) for dead | **0.266** | **0.712** | **0.879** | 0.790 | **yes** |
| 2026-11-11 | stoppage model, minimum runs re-picked held out | 0.268 | 0.715 | 0.883 | 0.790 | no: +0.002, folds disagree on the settings |
| 2026-11-11 | **+ acceleration**: 80th percentile of players' \|change of speed\|, ±3 s, as a fifth stoppage signal | **0.282** | **0.735** | **0.898** | 0.790 | **yes** |
| 2026-11-11 | a missing stoppage signal adds nothing instead of forcing dead (review of #22) | 0.282 | 0.735 | 0.898 | 0.790 | yes: same mean, the principled rule |

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
minutes macroF1 0.820, 15% dead against 20% true.

**Not every choice was made held out.** The weights are refitted per held-out provider, but which
signals to use (4 of 9 speed signals, then 1 of 6 acceleration signals) and their window lengths
were chosen on all 30 matches. The ablations say those choices move held-out accuracy by under
0.01; the technical document should still say so.

Before the acceleration signal, on BAR-ATH, as a check only: 101 / 108 possessions per half, mean 18.1 / 16.5 s, 43% / 44% dead;
public 5 minutes macroF1 0.814, spellF1 0.336, 8 possessions against 17. Fewer possessions and
more dead than before: the model now merges more, which the next work on turnovers should balance.

## v0: players only, no ball

The model never reads the ball. Tim's `hmm/` works from the ball, so the two see the match through
different eyes and should make different mistakes, which is what makes combining them worth it.

On each frame:
- **Who:** the more spread-out team has the ball. Spread is each team's mean distance from its own
  centre, averaged over 2 s either side of the frame. The keeper, taken as the team's deepest player
  toward its own goal, is left out: he adds several metres that depend on where play is, not on who
  has the ball.
- **Dead:** a stoppage model reads four players-only signals and calls the frame dead above even
  odds (`DEAD_SIGNALS`, `DEAD_FIT` in `model.py`):
  - median player speed over 0.5 s either side: the typical player walks at a stoppage
  - the 80th-percentile speed over the same window: in slow build-up the fastest few still run
  - median speed over 10 s either side: stoppages last, a slow patch in play doesn't
  - speed over the next 2 s minus the last 2 s: everyone slows into a stoppage and speeds up out
  - the 80th percentile of players' |change of speed| over 3 s either side: in play players keep
    changing pace as they react to the ball; at a stoppage they jog or walk steadily
  Its weights are a logistic regression on the 30 outside matches, each provider counted equally.
  The fitted weights (`DEAD_FIT["w"]`, after the intercept): median −0.58, 80th percentile
  −0.67, 10 s median −0.40, speed change −0.50, acceleration −1.35. All point the same way, and
  acceleration carries the most. A missing signal adds nothing either way. The viewer's Dead-ball
  model toggle shows each one's push for any frame. `dead=median` brings back the old rule (median under 1.2 m/s).
- **No blips:** within each half, alive/dead and the team lean are cleaned separately: an alive
  run under 2 s or a dead run under 3 s flips, and so does a home or away run of the lean under
  2 s, shortest first (`absorb_short`; `uv run python possession/model.py` checks it). Without
  this, a moment of walking inside a possession turned it D and split it in two. Cleaning the two
  apart means a dead blip can't move a turnover: the team changes where the lean does.
- Graded frames before each kickoff are D on BAR-ATH.

Windows look forward as well as back (the task is offline) and stop at the edge of each half.

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

What it gets wrong:
- **Missed turnovers.** Absorbing short runs fixed the over-splitting (520 possessions on BAR-ATH
  down to 255), and showed what it was hiding: the lean misses turnovers, so the model now makes
  about 30% too few possessions on BAR-ATH and about half the true count on the outside matches.
  Looser minimums won't recover them; a sharper turnover signal would.
- **Fixed minimums.** The 2 s / 3 s cutoffs suit this model's runs. They would need re-picking
  before being put on `hmm/` or a combined model, which have fewer short false runs.
- **Late changes.** Shape takes a couple of seconds to change after a turnover, and the 2 s window
  adds to that.
- **Style.** A team keeping the ball deep looks compact, and a high press looks stretched.
- **One number for who.** The team call is still the spread alone, with a fixed cutoff at zero.

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
