# stops/: possession around stoppages and restarts

Thread "Fix possession around stoppages", 2026-11-15. Base: the ball-taught model `bt`
(`deep/`, 0.545 raw / 0.626 folded at the shipped decode, mean of `bt` and `bts1`). Held out by
provider throughout; "folded" is the outside labels folded to the control rule (`external/control.py`).

**Outcome.** No players-only change beats `bt` at stoppages. One optional clue that is not
players-only, BAR-ATH's Smart Tagging set pieces, recovers stoppages the model plays through
(`deep/run.py --tags`, off by default).

## Where the score sits (`oracle.py`, `anatomy.py`)

Truth written into the model's probabilities, then the shipped decode (`oracle.py bt bts1`, raw / folded):

| written in | gain raw | gain folded |
|---|---|---|
| live or dead, every frame | +0.060 | +0.064 |
| live or dead, on true dead frames only (no missed stoppage frames) | +0.037 | +0.042 |
| live or dead, on true live frames only (no invented ones) | +0.023 | +0.023 |
| live or dead within 1 s of a stoppage's start or end | +0.023 | +0.016 |
| live or dead within 3 s | +0.042 | +0.041 |
| the full label within 3 s of a stoppage's start | +0.088 | +0.076 |
| the full label within 3 s of a restart | +0.053 | +0.051 |

At each true stoppage (`anatomy.py bt`):

| | IDSSE | Metrica | SkillCorner |
|---|---|---|---|
| stoppages over 5 s found | 90 to 98% | 84 to 99% | 90 to 97% |
| stoppages of 2 to 5 s found (7, 9, 4 a match) | 53% | 67% | 64% |
| start within 0.5 s | 36% | 40% | 24% |
| end within 0.5 s | 52% | 51% | 56% |
| team right in the last 2 s before the stop | 42% | 56% | 55% |
| a possession under 2 s ends at the stop | 40% of stops | 33% | 22% |
| team right in the first 2 s after a throw-in / goal kick / free kick | 89 / 92 / 94% | 75 / 67 / 77% | 71 / 73 / 64% |

So the model finds the stoppages; what it loses is the last short possession before the ball
goes dead (a clearance or a tackle into touch, the short-turnover problem), the half second
either side, and on the two providers whose labels come from events, who restarts.

## Tried, held out

| | raw | folded | |
|---|---|---|---|
| `bt` | 0.545 | 0.626 | |
| `btr`: bt taught, in training only, who takes each restart (pointer at the taker for the 6 s before play resumes), the kind of restart, and the seconds to and since a stop or start (`deep/restarts.py`, `train.py --restart 0.5`), one seed | 0.522 | 0.614 | no: IDSSE -0.032 (0 of 7 matches up), Metrica -0.038 (0 of 3), SkillCorner +0.001. Stoppage timing and restart team unchanged; the team split got worse. Second seed stopped at step 800 |
| bt's team split on the dead-ball probability averaged over bt and bts1 (`mixdead.py`) | 0.549 | 0.631 | no: +0.004, seed averaging |
| the same over six or seven nets (bt, bts1, btk, btc, v2, v2s1, btr) | 0.548 | 0.630 | no |
| bt's team split on btr's dead-ball probability | 0.542 | 0.623 | no |

Ruled out without a run:

- **Ball visibility as a live/dead label on BAR-ATH.** On the Metrica sample matches the ball is
  seen only in live play (99.9%), but on BAR-ATH's public window 15% of ball-seen frames are dead,
  and writing "seen means live" into bt there lowers it (0.745 to 0.723). It would also be the ball.
- **Smart Tagging phase teams.** In the public window a PROGRESSION phase names the team in
  possession on 72% of its frames; bt agrees with the phases on 90% over the match. Too loose to
  correct the team.

## The tag clue (`tags.py`, `deep/run.py --tags`)

bt plays through 4 of Barcelona's 7 tagged goal kicks and one tagged throw-in: a goalkeeper with
the ball looks the same from the players whether it is a goal kick or open play, and its dead-ball
probability sits at 0.2 to 0.4 for the whole stretch. Where a tagged set piece (not a kick-off)
has no decoded stoppage from 3 s before to 6 s after the tag opens, the dead-ball odds are
multiplied by 4 from 20 s before to 6 s after, and the half is decoded again.

- Public 5 minutes (one such tag, check only): spellF1 0.745 to 0.786, macroF1 0.904 to 0.961;
  the drawn stoppage is 18025 to 18420 against the truth's 18027 to 18389. Odds of 2, 4, 8 and 20
  give the same spellF1.
- Whole match: 695 of 158,857 graded frames change; a stoppage is drawn at 3 of the 5 tags
  (the other two stay as they were at odds 4).
- It cannot be scored held out: no outside match has Smart Tagging. The tags are automatic
  (FAQ.md allows them) but they are not the players, so this is off unless asked for.

```bash
python stops/oracle.py bt bts1
python stops/anatomy.py bt
python stops/mixdead.py bt bt,bts1
sh deep/queue.sh stops/jobs.txt            # btr (and its second seed, not finished)
python deep/paired.py bt btr
python deep/run.py --tag bt --tags --out deep_bt_tags.csv
```
