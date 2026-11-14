# adapt/: BAR-ATH's tracking, and a missing-player test that copies it

Works on any players-only base model whose fold nets are saved in the `data/cache/deep/<tag>/<fold>/`
layout. Reads players only. Independent of `hmm/` and `seq/` (the folded labels used as a second
yardstick come from `external/control.py`, which is scoring, not a model input).

```bash
python adapt/dropout.py --fit --check                    # measure BAR-ATH, calibrate, compare
python adapt/test.py --tag v2 --cond full old bar bar+fill15
python adapt/bar_ath.py --tag v2 --fill 15               # what filling changes on BAR-ATH itself
```

## How BAR-ATH's tracker loses players (tracking alone, no labels)

- 7.6% of player-frames are missing. A track ends 1,198 times; the player is then gone for
  seconds (almost no gap is under 1 s).
- It is occlusion, not chance. Losses per player-minute by situation:

| nearest other tracked player | under 0.5 m/s | 0.5 to 1.5 | 1.5 to 3 | over 3 m/s |
|---|---|---|---|---|
| under 0.75 m | 12.9 | 4.5 | 1.5 | 0.16 |
| 0.75 to 1.5 m | 6.9 | 2.9 | 1.1 | 0.28 |
| 1.5 to 3 m | 2.0 | 0.77 | 0.41 | 0.11 |
| 3 to 6 m | 0.51 | 0.24 | 0.06 | 0.02 |
| over 6 m | 0.51 | 0.26 | 0.04 | 0.007 |

- The frames with 14 or fewer tracked (5.3%) are 33 stretches bunched inside a few stoppages (after
  the goals, at the end of the first half). The ball is never seen in them and the players are
  walking: huddles, where players hide each other.
- Not the cause: the camera's view (players well outside the tactical picture are tracked) and
  position noise (BAR-ATH is as smooth as Metrica and SkillCorner; only IDSSE is rougher).
- Also there, not simulated: more than 22 tracks on 2.4% of frames, in stoppages.

`external/score.py drop_players` (the test behind "0.516 with players missing") loses every player
at one flat rate and drops its bursts anywhere, so 64% of its frames with under 20 tracked are
live play. `dropout.simulate` uses the table above on each visible player, keeps him lost at least
1 s, and brings him back slowly while he stands or walks and quickly once he moves. Those two
return rates are calibrated, not measured (a lost player's situation is unseen):

| | player-frames missing | 14 or fewer tracked | under 20 | all 22 |
|---|---|---|---|---|
| BAR-ATH | 0.076 | 0.053 | 0.149 | 0.518 |
| simulated on IDSSE (7) | 0.076 | 0.047 | 0.215 | 0.413 |
| simulated on Metrica (3) | 0.066 | 0.023 | 0.174 | 0.370 |
| simulated on SkillCorner (20, frames it tracks) | 0.037 | 0.010 | 0.080 | 0.582 |

The simulation has somewhat more frames with one or two lost and fewer total collapses than
BAR-ATH; SkillCorner comes out milder, its players being placed further apart in stoppages.

## The test (2026-11-14, `deep` v2 fold nets, fixed decode, held out by provider)

spellF1; raw labels with cost 4 and 1 s / 1 s runs, folded labels with 2 s / 3 s.

| held-out tracking | labels | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|---|
| as recorded | raw | 0.549 | 0.635 | 0.398 | 0.527 |
| | folded | 0.672 | 0.723 | 0.489 | 0.628 |
| old knock-out (`old`) | raw | 0.543 | 0.620 | 0.392 | 0.519 |
| | folded | 0.664 | 0.706 | 0.480 | 0.617 |
| lost as BAR-ATH loses them (`bar`) | raw | 0.546 | 0.635 | 0.396 | 0.526 |
| | folded | 0.666 | 0.720 | 0.486 | 0.624 |
| `bar`, gaps to 5 s filled | raw / folded | | | | 0.526 / 0.624 |
| `bar`, gaps to 15 s filled | raw | 0.547 | 0.637 | 0.398 | 0.527 |
| | folded | 0.669 | 0.722 | 0.490 | 0.627 |
| `bar`, gaps to 60 s filled | raw / folded | | | | 0.528 / 0.627 |

- Losing players the way BAR-ATH does costs 0.001 (raw) to 0.004 (folded), not the 0.008 to 0.013
  the old test charged. The players it loses are mostly standing in a stoppage, where the answer
  is "dead" whoever is visible.
- In the simulation 21% of graded frames with under 20 tracked are live play, and the model calls
  24% of those dead (4% of other live frames). That is the remaining exposure: about 3% of
  BAR-ATH's frames if the same share holds there, which cannot be checked without labels.
- Filling gaps (`impute.fill`, straight line between where a track ends and where the nearest
  reachable track of the same team starts) wins back what there is: +0.001 raw, +0.003 folded at
  15 s. That is below the +0.01 bar, so it is not adopted as a gain. It is harmless and can sit in
  front of whichever base model wins. On BAR-ATH itself (`bar_ath.py`), filling to 15 s changes
  1.5% of v2's labels, mostly in frames with under 20 tracked; the public 5 minutes are unchanged
  at spellF1 0.826 (they have no such frames).

## The unseen-provider cost is not about missing players

Five matches are scored both by nets that trained on their provider and by the net that did not:

| | spellF1 raw | spellF1 folded | alive/dead F1 | team accuracy |
|---|---|---|---|---|
| provider in training | 0.527 | 0.614 | 0.955 | 0.868 |
| provider unseen | 0.470 | 0.558 | 0.946 | 0.846 |

Folding the labels does not shrink it (0.057 and 0.056), so it is not only the providers' label
conventions. It is uneven: the Metrica match loses nothing unseen (0.621 against 0.630 / 0.654 raw,
0.723 against 0.716 / 0.723 folded), IDSSE loses 0.06 and SkillCorner 0.07. BAR-ATH's tracking was
made by Metrica Sports (its ATD metadata says so) and is as smooth as the Metrica matches, which
is mild grounds to expect the low end of that cost, from one match.
