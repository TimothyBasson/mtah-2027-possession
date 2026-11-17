# turnover/: a dedicated detector for turnovers the model misses

Question (2026-11-15): the ball-taught model `bt` rarely finds short possessions. Can a detector
built only for change-of-possession moments (tackles, interceptions, loose balls won) find them
from the players' tracking and hand them to the decode?

**Answer: not enough to adopt.** The detector is worth about +0.01 on IDSSE and Metrica and
nothing on SkillCorner, over two seeds, below the bar (+0.01 mean, no provider down). Nothing in
`deep/` or in what labels BAR-ATH changes. What was learned is below.

Players only at prediction, as everywhere: the ball and the labels are training targets. Base
model: `bt` + `bts1` averaged, shipped decode (cost 4, 1 s minimums). Held out by provider.
"Raw" is the labels as given, "folded" the labels folded to the control rule.

## 1. What a turnover detector could be worth (`oracle.py`)

The true direct H<->A change times handed to the decode of the held-out bt probabilities:

| | IDSSE | Metrica | SkillCorner | mean raw | mean folded |
|---|---|---|---|---|---|
| base | 0.584 | 0.649 | 0.404 | 0.546 | 0.628 |
| true times as a discount on switching | 0.619 | 0.681 | 0.409 | 0.570 | 0.642 |
| true times, switching elsewhere made dear | 0.669 | 0.719 | 0.436 | 0.608 | 0.676 |
| the same from a detector with 70% recall | 0.629 | 0.683 | 0.429 | 0.581 | 0.665 |
| 70% recall and 2 false alarms a minute | 0.628 | 0.681 | 0.426 | 0.578 | 0.654 |

So the whole angle has a ceiling near +0.05 to +0.06, and SkillCorner's score is not held down
by turnovers alone.

## 2. What the base model finds today (`recall.py`, `offsets.py`)

True turnovers matched by one of the model's own, same direction, within 1 s:

| | recall | precision | recall where a neighbouring possession is under 1 s / 1-2 s / 2-5 s / over 5 s |
|---|---|---|---|
| IDSSE | 0.38 | 0.71 | 0.13 / 0.28 / 0.53 / 0.63 |
| Metrica | 0.43 | 0.73 | 0.16 / 0.21 / 0.50 / 0.64 |
| SkillCorner | 0.20 | 0.20 | 0.05 / 0.11 / 0.24 / 0.33 |

On SkillCorner the model's turnover times are close to unrelated to the labelled ones. Its
labels are also loosely tied to the ball: the winning team's nearest touch lies within a quarter
second of the labelled turnover 19% of the time, against 75% on IDSSE and 62% on Metrica.

## 3. The detector (`tdata.py`, `tnet.py`, `ttrain.py`)

Unlike `deep/`, players keep their identity through a half at 12.5 fps (velocity and
acceleration over 0.24 s). Three blocks alternate a convolution along each player's own track
with attention across the players of the frame; the net then points at the player on the ball
and says per player whether he touches it (both taught by the real ball on training matches),
and frame heads say "the ball passes to home / to away here" for the labels as given (`ev`) and
folded (`evf`). After Kim et al. (KDD 2023) and PathCRF (arXiv 2602.12080, possessor paths from
players only, 70% edge accuracy in play). 0.87 M weights, 1,200 steps of 16 ten-second windows,
about 40 minutes a fold on one thread.

As a turnover finder, held out (`detect.py`, within 1 s, same direction):

| | base recall / precision | detector above 0.1 | above 0.2 | of the base's misses found (0.1) |
|---|---|---|---|---|
| IDSSE | 0.38 / 0.71 | 0.70 / 0.39 | 0.40 / 0.63 | 54% |
| Metrica | 0.43 / 0.73 | 0.70 / 0.37 | 0.42 / 0.59 | 53% |
| SkillCorner | 0.19 / 0.20 | 0.37 / 0.14 | 0.27 / 0.17 | 25% |

At the base's precision the detector has the base's recall: a different architecture at 2.5
times the frame rate reads the same turnovers out of the players. It can be pushed to 70% recall
only at 39% precision (about 6 false alarms a minute), which section 1 prices at about +0.01.
The direction is almost never wrong (1% of peaks).

Its false alarms on IDSSE (`alarms.py`): 24% the other team keeps the ball and the named team
never touches it (pressure that does not win it), 16% a true turnover 1 to 2 s away, 9% beside a
stoppage, 11% the named team already had it. On SkillCorner 56% of peaks are beside a stoppage.

## 4. In the decode (`apply.py`)

A switch to the team the detector names costs less by w times its log-odds above a floor.
One setting in all folds, change against the base, raw / folded:

| detector | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| t1 (seed 0), best setting read off the table | +0.010 / +0.007 | +0.010 / +0.002 | -0.001 / -0.004 | 0.552 / 0.630 |
| t1s1 (seed 1), best setting | +0.009 / +0.004 | +0.009 / +0.000 | -0.001 / -0.005 | 0.551 / 0.628 |
| both averaged, best setting | +0.009 / +0.006 | +0.009 / -0.000 | -0.001 / -0.004 | 0.551 / 0.629 |
| t1, setting picked on validation matches | -0.004 / +0.003 | -0.006 / -0.003 | +0.004 / +0.006 | 0.544 / 0.630 |

The best rows are chosen on the test table, so they flatter. Picked honestly the gain is zero.
Switching the evidence off within a second of a stoppage changes nothing. The detector's own
H/A/D side head decoded alone scores 0.471 / 0.519 / 0.373 and averaging it into the base lowers
the base.

## 5. Ceiling: the same detector shown the ball (`ttrain.py --ball-input`, analysis only)

To see how much of a turnover is in the players' movement at all, the same net was trained with
the recorded ball as one more token. This net reads the ball, so it labels nothing; it is a
measurement. Average precision of "a turnover within 0.24 s", one match each:

| held-out fold | unseen match of a trained provider: players only -> with ball | held-out provider: players only -> with ball |
|---|---|---|
| IDSSE | 0.20 -> 0.90 (Metrica match) | 0.21 -> 0.15 |
| Metrica | 0.33 -> 0.55 (IDSSE match) | 0.19 -> 0.49 |
| SkillCorner | 0.36 -> 0.59 (IDSSE match) | 0.07 -> 0.08 |

Two readings. Within a provider the ball lifts turnover detection from 0.2-0.36 to 0.55-0.90:
most of what marks a turnover is the ball changing feet, and the players' movement carries a
third of it or less. Across providers even the ball does not carry over (IDSSE 0.15, SkillCorner
0.08): where a provider puts a turnover is its own convention, which fits the earlier finding
that the sources differ in how they cut possessions, not in how they track.

## Conclusion

- Short turnovers missing from bt are missing because the players alone barely show them, not
  because of the network, its frame rate or its loss. Two different networks agree on which
  turnovers are visible.
- The moments a ball-blind model cannot see are the ones a ball-reading model sees best, so this
  is the part of the score to expect from the combination with the ball model at the end, not
  from more players-only work.
- Not tried: turnover targets snapped to the ball touch (would not change the labels scored
  against); a detector at 25 fps; training the detector on BAR-ATH-like control-rule labels only.

```bash
python turnover/oracle.py; python turnover/recall.py; python turnover/offsets.py
sh turnover/queue.sh t1            # three folds; --seed 1 for t1s1; --ball-input for the ceiling
python turnover/detect.py --det t1; python turnover/alarms.py --det t1
python turnover/apply.py --det t1  # needs pip install numba for speed; same result without
```

Fold nets and held-out outputs: `/mnt/project-files/turnover/{t1,t1s1,tball}/<fold>/`.
