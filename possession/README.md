# possession/: Jaedon's possession model

This model is separate from `hmm/`. The plan is to compare and combine the two at the end.

```bash
uv run python possession/run.py                    # -> evaluation/results/possession.csv (BAR-ATH)
uv run python external/score.py --seen 0.59        # score on the 10 outside matches
uv run python viz/combined/make_combined.py --match all --explain --open   # see how it works
```

## v0: how it works

The simplest model. On each frame, the team whose nearest player is closer to the ball has it.
Where the ball isn't seen, the label is D. There is no smoothing.

| outside matches, `--seen 0.59` | spellF1 | macroF1 |
|---|---|---|
| IDSSE (7) | 0.041 | 0.432 |
| Metrica (3) | 0.096 | 0.767 |
| all 10 | 0.057 | 0.532 |

What it gets wrong:
- **Far too many possessions.** About 5 times too many (3,609 against 614 on idsse_J03WMX), with half of them under 1 s. The label flips every time the two nearest players swap.
- **A missing ball counted as dead.** On Metrica, and on our match by the players' speed, a missing ball usually is dead. IDSSE's dropout is random, so there it is wrong half the time.

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
