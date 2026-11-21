# story/: possession story timeline and turnover autopsy

A first sketch for the Part 2 Innovation Challenge: one page that tells BAR-ATH as a sequence of
possession spells and takes apart every ball lost in open play.

```bash
python story/make_story.py                 # -> data/viz/story.html (about 2 MB, LOCAL ONLY: has the tracking)
python story/make_story.py --no-tracking --out data/viz/story_shared.html   # 0.3 MB, no replay: the one to share
python story/make_story.py --labels other.csv --json /tmp/story.json
```

Needs numpy, scipy and kloppy (for `possession/load.py`), plus the BAR-ATH files in `data/tactical/`.
Runs in about 10 s. Default labels: the file we submitted (`final/Gambtt_submitted.csv`).

## What is on the page

- **Every possession.** The whole match as spells (runs of H or A in the labels), Barcelona above,
  Athletic below, open-play turnovers ticked. Click a spell to replay it.
- **Replay.** A top-down pitch at 5 Hz (interpolated) from 3 s before the spell to 6 s after it, with
  a ball trail. Opened from `data/viz/` on the laptop it also finds `../tactical/BAR-ATH_tactical.mp4`
  and shows the video in sync.

**Keep the full page local.** It embeds every slot and the ball at 5 Hz, and RULES.md forbids
publishing derived data that would reconstruct the tracking ("frame-level tracking exports"). Anything
shared, with judges or as a link, is built with `--no-tracking`: same timeline, summaries, autopsy,
patterns and loss map, and the pitch shows only where each spell started and ended. Ask the
organisers before sharing anything with the tracking in it.
- **Spell card.** A generated summary (how it was won: open play or which restart from Smart
  Tagging; how far the ball got; how it ended; tagged moments), plus the autopsy when it ended in a
  turnover.
- **Turnover autopsy.** For each open-play loss:
  - where (thirds and lanes in the losing team's direction of attack) and the action in the 2 s
    before (ball movement: back pass, square, forward pass, long forward ball, carry; Smart Tagging's
    long ball wins)
  - shape 5 s before (or at the start of a shorter spell) against the team's average on the ball:
    height, width, length beyond one standard deviation
  - **out of position**: each team's usual shape with and without the ball is learned from this
    match (players matched to 10 role slots frame by frame, Hungarian assignment around the team
    centre). A player is flagged when further from their role's spot than 9 times in 10 and more than
    8 m. The page draws the usual spots as ghosts at that moment. Tracker slots are not named
    players, so roles stand in for names. Football names come from the shape without the ball; roles
    with the ball take the name of the role the same tracker slot holds after the turnover.
  - after the loss: players within 10 m of the ball at +2 s, seconds until the team's centre starts
    moving back (1 s velocity under -0.5 m/s towards its own goal), won back within 5 / 10 s, and
    whether the opponent's spell reached the final third or produced a chance tag
  - **recurring patterns**: turnovers grouped by zone, zone and action, action, quick loss, shape
    and the most out-of-position role; groups of 3 or more, most frequent first, each with the share
    where the opponent then reached the final third

Left and right are checked against Smart Tagging's "Side" on final-third entries (32 of 33 agree).

## Caveats

Short `X` gaps (up to 60 s) carry the previous label forward before spells are cut, as the scorer
does (`submission/make_submission.fill_short_abstentions`); the submitted file has none inside the
halves, so its counts do not change.

Spell lengths are model output. "Lost within 5 s of winning it" and "won back within 5 s" rest on
short spells, the model's weak spot. In the public 5 minutes the file draws 4 spells under 5 s
where the truth has 6, so it does not overcount them there, but 5 minutes is a small sample and
which short spells it draws is less certain. Quote those numbers as model output.

Possession comes from our model, not events, so some short spells are wrong (the model's weak
spot). The 4 s team reaction found earlier (alldata/react/) is measured differently: there, the
direction of both teams' movement; here, the first moment the losing team's centre runs back,
which comes earlier (median 1.6-1.8 s).
