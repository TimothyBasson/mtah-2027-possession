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
| 2026-11-12 | **three signals**: drop the 80th-percentile speed and the 10 s median (table below) | **0.282** | **0.736** | **0.898** | 0.790 | **yes**: as good with two fewer |
| 2026-11-12 | **1 s lean window, 1 s minimum possession** (were 2 s, 2 s), from a football-prior grid (below) | **0.287** | **0.736** | **0.899** | 0.790 | **yes**: every provider up, on a flat plateau |
| 2026-11-12 | **team model**: logistic on spread, width and forward velocity instead of spread alone (below) | **0.333** | **0.757** | **0.899** | **0.824** | **yes**: every provider up |

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

## Team model (2026-11-12)

The breakdown below showed half the mistakes are the team, and the sweep showed smoothing can't
fix them, so the team call became a small fitted model like the stoppage one. Candidate signals
each needed a football reason first, and were scored frame by frame held out (logistic fitted on
alive frames of two providers, scored on the third, providers equal; accuracy on all alive frames
and within 2 s of a true turnover):

| signals | accuracy | near a turnover | kept? |
|---|---|---|---|
| spread alone (as before) | 0.782 | 0.591 | |
| + **forward velocity**: mean player x velocity toward the goal home attacks, ±0.4 s, smoothed 1 s. The team with the ball pushes up, the other drops | 0.806 | 0.610 | **yes**: every provider up, the same for any lag (0.4 / 1 s) or smoothing (0.5-2 s) |
| + territory: where the block is past halfway | 0.809 | 0.617 | no: came out with the team in possession deeper, no clear football reason for that sign |
| + goal-side marking: share of a team's players with an opponent within 5 m goal-side | 0.804 | 0.611 | no: 0.52 alone, nothing added |
| + team length, home minus away | 0.811 | 0.618 | no: negative weight, no clear football reason |
| + **width**, home minus away: make the pitch big with the ball, small without it | 0.816 | 0.614 | **yes**: every provider up |

No intercept and no centring: nothing in the game favours home, so zero on every signal is even
odds. Fitted on all 30 (`TEAM_FIT`): spread 0.455, width 1.381, forward velocity 0.465 per sd.
Held out with the 1 s settings: spellF1 IDSSE 0.243 → 0.294, Metrica 0.317 → 0.372, SkillCorner
0.301 → 0.333, mean 0.287 → 0.333; team accuracy 0.790 → 0.824; macroF1 0.736 → 0.757.

Each left out, refitted, held out:

| left out | spellF1 | macroF1 | team |
|---|---|---|---|
| nothing | 0.333 | 0.757 | 0.824 |
| spread | 0.334 | 0.757 | 0.823 |
| width | 0.322 | 0.751 | 0.815 |
| forward velocity | 0.307 | 0.749 | 0.811 |
| width and velocity (spread alone) | 0.287 | 0.736 | 0.790 |
| spread and width (velocity alone) | 0.203 | 0.625 | 0.603 |

Spread adds nothing once width is in (SkillCorner 0.330 without it against 0.333 with it, the
other two slightly up). It is kept anyway: it is a mean over every outfield player, so it holds up
when part of a team is out of the camera's view, which width (a max minus a min) does not, and
BAR-ATH is filmed by a camera that sees part of the pitch. Dropping it is a one-line change if the
next work makes it look dead weight.

Breakdown with the team model (`external/errors.py`, means): wrong frames 25.4% → 23.5%; wrong
team mid-possession 5.1% → 3.6%, near a turnover 7.2% → 6.8%; true possessions matched 21% → 26%,
merged 34% → 31%, predicted per true 0.62 → 0.68; true turnovers found within 10 s 41% → 49%,
still 1-2 s late (median). Turnovers remain the weak spot: every signal here is a shape or a
drift, and none reacts at the moment of the turnover itself.

On BAR-ATH (check only): 257 possessions of mean 14.9 s (was 223 of 17.2 s); public 5 minutes
macroF1 0.825 (was 0.824), spellF1 0.486 (was 0.352), 11 possessions against 17 (was 8).

## Where the mistakes are (2026-11-12)

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

## v0: players only, no ball

The model never reads the ball. Tim's `hmm/` works from the ball, so the two see the match through
different eyes and should make different mistakes, which is what makes combining them worth it.

On each frame:
- **Who:** a team model reads three players-only signals, each home minus away and averaged over
  1 s either side, and calls home above even odds (`TEAM_SIGNALS`, `TEAM_FIT` in `model.py`):
  - spread, each team's mean distance from its own centre: attackers stretch, defenders stay
    compact. The keeper, taken as the team's deepest player toward its own goal, is left out: he
    adds several metres that depend on where play is, not on who has the ball.
  - width across the pitch: make the pitch big with the ball, small without it
  - forward velocity, every player's x velocity toward the goal home attacks: the team with the
    ball pushes up, the other drops. Which goal is whose is read per half from where each team stands.
  Its weights are fitted like the stoppage model's, with no intercept (team section above).
- **Dead:** a stoppage model reads three players-only signals and calls the frame dead above even
  odds (`DEAD_SIGNALS`, `DEAD_FIT` in `model.py`):
  - median player speed over 0.5 s either side: the typical player walks at a stoppage
  - speed over the next 2 s minus the last 2 s: everyone slows into a stoppage and speeds up out
  - the 80th percentile of players' |change of speed| over 3 s either side: in play players keep
    changing pace as they react to the ball; at a stoppage they jog or walk steadily
  Its weights are a logistic regression on the 30 outside matches, each provider counted equally.
  The fitted weights (`DEAD_FIT["w"]`, after the intercept): median −1.46, speed change −0.48,
  acceleration −1.58. All point the same way, and acceleration carries the most. The
  80th-percentile speed and 10 s median were dropped once acceleration made them redundant (above). A missing signal adds nothing either way. The viewer's Dead-ball
  model toggle shows each one's push for any frame. `dead=median` brings back the old rule (median under 1.2 m/s).
- **No blips:** within each half, alive/dead and the team lean are cleaned separately: an alive
  run under 1 s or a dead run under 3 s flips, and so does a home or away run of the lean under
  1 s, shortest first (`absorb_short`; `uv run python possession/model.py` checks it). Without
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
- **Fixed minimums.** The 1 s / 3 s cutoffs suit this model's runs. They would need re-picking
  before being put on `hmm/` or a combined model, which have fewer short false runs.
- **Late changes.** Shape takes a couple of seconds to change after a turnover: the model calls
  the turnovers it finds 0.6-2 s late, and finds only about 40% within 10 s (`external/errors.py`).
- **Style.** A team keeping the ball deep looks compact, and a high press looks stretched.

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
