# Keeper-area ball clue (hmm/keeper.py, `hmm.decode --keeper M`)

Target: false stoppages where the keeper holds the ball after a save or a gathered cross. On
BAR-ATH the tracker loses the ball in the keeper's hands, the players slow down, and the decoder
reads dead ball.

The look-ahead clue (hmm/ahead.py, `--ahead`, off by default) reads every missing-ball gap by where
the ball turns up next; at 4x it removed some of these but cost held-out score, because its line and
open-play categories move stoppages elsewhere. This clue keeps only the keeper part:

* a missing-ball gap counts for a keeper only when the ball was **lost in play inside his own box**
  and **turns up in his hands** (within 2 m of him, inside his box), with neither end on or past a
  line ("box" mode; "next" drops the lost-in-his-box condition). Without the line condition, on
  the external matches "ball missing at the keeper" is mostly a ball out for a goal kick, and the
  counted table says dead ball (logs_kp/sweep1.log: 0.6662 to 0.6697 at 4x);
* every other frame gets no term.

The table is p(category | state) counted on the external labels, each provider equal (own keeper
vs dead ball about +1.2 in log ratio); when scoring held out, each provider uses a table counted
on the other two only.

## Held out (kp/score_keeper.py, restart hold on, bt+b2 nets, as v9)

| keeper weight | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| 0 (v9) | 0.726 | 0.807 | 0.496 | 0.6764 |
| box 4x | 0.726 | 0.807 | 0.496 | 0.6763 |
| box 8x | 0.726 | 0.805 | 0.495 | 0.6755 |
| box 16x | 0.726 | 0.806 | 0.495 | 0.6756 |
| next 8x | 0.726 | 0.803 | 0.495 | 0.6748 |
| box 4x + ball seen still at keeper | 0.722 | 0.807 | 0.495 | 0.6747 |

Fair pick of the weight per provider on the other two is 0 everywhere, so the fair number stays
v9's 0.663. The scorer never hides the ball in a keeper's hands, so the held-out yardstick cannot
show the effect it is aimed at; it shows the clue costs little.

## BAR-ATH (kp/barath.sh)

On by default at 4x since 2026-11-17 (hmm_v9.csv sha256 6c565c68...5eefa121e). `--keeper 0` gives the file of PR 60 (sha256 e76d697a...bedc7221ed). Public window spellF1 0.774
at every weight. At 4x, 0.9% of graded frames change and the dead share goes 0.470 to 0.462; at 8x,
1.4% and 0.457.
