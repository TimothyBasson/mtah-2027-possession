# possession/: Jaedon's possession model

This model is separate from `hmm/`. The plan is to compare and combine the two at the end.

```bash
uv run python possession/run.py                    # -> evaluation/results/possession.csv (BAR-ATH)
uv run python external/score.py --seen 0.59        # score on the 10 outside matches
uv run python viz/combined/make_combined.py --match all --explain --open   # see how it works
```

## v1: how it works

1. **Clues.** Each clue scores every frame for H, A and D (`model.clues`):
   - **nearest:** which team's nearest player is closer to the ball.
   - **ball out:** the ball is seen more than 1 m past a line.
   - **player speed:** the mean player speed. Players jog or run in play and walk at a stoppage.
2. **Sum.** The clue scores are added and multiplied by `weight`. Neighbouring frames repeat the same look, so a single frame should not count for much.
3. **Smooth.** Viterbi picks the best label sequence per half. Every change of label costs `team_cost` (H↔A) or `dead_cost` (to or from D), so a short flicker isn't worth a new possession.

All numbers are in `model.DEFAULTS`. The two costs were picked by a sweep on the outside matches, never on BAR-ATH.

| outside matches, `--seen 0.59` | spellF1 | macroF1 |
|---|---|---|
| IDSSE (7) | 0.269 | 0.723 |
| Metrica (3) | 0.438 | 0.746 |
| all 10 | 0.320 | 0.730 |

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
