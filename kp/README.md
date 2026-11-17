# Keeper-area ball clue (hmm/keeper.py, `hmm.decode --keeper M`)

Goal: the BAR-ATH notes where the keeper holds the ball after a save or a gathered cross and v9
calls a stoppage. BAR-ATH's tracker loses the ball in the keeper's hands, the players slow down,
and the decoder reads dead ball.

The look-ahead clue (hmm/ahead.py, `--ahead`) fixed some of these at 4x but cost held-out score,
because its line and open-play categories also move stoppages elsewhere. This clue keeps only
the keeper part:

* a missing-ball gap counts for a keeper only when the ball was **lost in play inside his own
  box** and **turns up in his hands** (within 2 m of him, inside his box), with neither end on or
  past a line. That last condition matters: without it, on the external matches "ball missing at
  the keeper" is mostly a ball out for a goal kick, and the counted table says dead ball
  (sweep1: 0.6662 to 0.6697 at 4x);
* every other frame gets no term at all.

The table is p(category | state) counted on the external labels (each provider equal; held out by
provider when scoring). Own keeper vs dead ball is about +1.2 in log ratio.

## Held out (kp/score_keeper.py, restart hold on, bt+b2 nets, as v9)

| keeper weight | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| 0 (v9) | 0.726 | 0.807 | 0.496 | 0.6764 |
| box 4x | 0.726 | 0.807 | 0.496 | 0.6763 |
| box 8x | 0.726 | 0.805 | 0.495 | 0.6755 |
| box 16x | 0.726 | 0.806 | 0.495 | 0.6756 |
| next 8x (ball turns up at keeper, lost anywhere in play) | 0.726 | 0.803 | 0.495 | 0.6748 |
| box 4x + ball still at keeper | 0.722 | 0.807 | 0.495 | 0.6747 |

Fair pick of the weight per provider on the other two is 0 everywhere, so the fair number stays
v9's 0.663. The yardstick cannot reward this clue: the scorer never hides the ball in a keeper's
hands. It is a BAR-ATH fix judged on the notes.

## BAR-ATH (kp/barath.sh, kp/notes.py)

`--keeper 0` is byte-identical to main (sha256 e76d697a...bedc7221ed). Public window spellF1 0.774
at every weight (no keeper hold inside it).

Keeper notes, seconds of D in the 10 s after the note, v9 -> box 4x / box 8x:

| note | v9 | 4x | 8x |
|---|---|---|---|
| 15:57.8 | 6.2 | 6.2 | 6.2 (ball turns up 14 m from the keeper: he has already released it) |
| 17:47.5 | 10.2 | 10.2 | **0** |
| 46:13.0 | 2.6 | 2.6 | 2.6 (ball lost outside the box; `next` mode fixes it at a higher held-out cost) |
| 50:58.5 | 9.8 | **0** | **0** |
| 54:49.9 | 10.0 | **0** | **0** |
| 82:19.2 | 10.0 | 10.0 | 10.0 (no keeper gap there) |
| 89:17.1 | 9.8 | 1.8 | **0** |
| 112:12.4 | 1.9 | **0** | **0** |

Changes away from the notes at 8x: 82:58 (7.9 s), 116:55 (4.2 s) and 118:39 (5.8 s), D to the away
team. All three are a ball lost near the away keeper and found in his hands, with Smart Tagging
showing Barcelona in the box and then Athletic building up; they look like unnoted gathers, but
nobody has checked them on video. Three sub-2 s H>A flips at 19:57, 26:57, 37:00.
