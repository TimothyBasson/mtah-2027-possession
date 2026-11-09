# possession/: Jaedon's possession model

This model is separate from `hmm/`. The plan is to compare and combine the two at the end.

```bash
uv run python possession/run.py                    # -> evaluation/results/possession.csv (BAR-ATH)
uv run python external/score.py --seen 0.59        # score on the 10 outside matches
uv run python viz/combined/make_combined.py --match all --explain --open   # see how it works
```

## v0: players only, no ball

The model never reads the ball. Tim's `hmm/` works from the ball, so the two see the match through
different eyes and should make different mistakes, which is what makes combining them worth it.

On each frame:
- **Who:** the more spread-out team has the ball. Spread is each team's mean distance from its own
  centre, averaged over 2 s either side of the frame. The keeper, taken as the team's deepest player
  toward its own goal, is left out: he adds several metres that depend on where play is, not on who
  has the ball.
- **Dead:** when the median player's speed, averaged over 0.5 s either side, is under 1.2 m/s.
- Graded frames before each kickoff are D on BAR-ATH.

Windows look forward as well as back (the task is offline) and stop at the edge of each half.

| outside matches | spellF1 | macroF1 | alive/dead | team |
|---|---|---|---|---|
| IDSSE (7) | 0.187 | 0.667 | 0.831 | 0.760 |
| Metrica (3) | 0.212 | 0.660 | 0.829 | 0.794 |
| all 10 | 0.194 | 0.665 | 0.830 | 0.771 |

With the keepers counted (`external/score.py keepers=1`) it was 0.182 / 0.655 / 0.830 / 0.752.
Leaving them out raised team accuracy on 9 of the 10 matches.

For comparison, the earlier nearest-player-to-ball v0 scored spellF1 0.057 and macroF1 0.532 on all
10 (`--seen 0.59`). `--seen` does nothing here, because the ball is never read.

Possessions now come out about the right length: 634 predicted against 614 true on idsse_J03WMX
(median 4.8 s against 4.2 s), and 680 against 439 on metrica_1.

What it gets wrong:
- **Late changes.** Shape takes a couple of seconds to change after a turnover, and the 2 s window
  adds to that.
- **Style.** A team keeping the ball deep looks compact, and a high press looks stretched.
- **One number per question.** Each of who and dead uses a single signal with a fixed cutoff.

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
