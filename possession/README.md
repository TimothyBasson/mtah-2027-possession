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
- **No blips:** within each half, any H/A run under 2 s or D run under 3 s is relabelled as its
  longer neighbour, shortest first (`absorb_short`; `uv run python possession/model.py` checks it).
  Without this, a moment of walking inside a possession turned it D and split it in two.
- Graded frames before each kickoff are D on BAR-ATH.

Windows look forward as well as back (the task is offline) and stop at the edge of each half.

| outside matches | spellF1 | macroF1 | alive/dead | team |
|---|---|---|---|---|
| IDSSE (7) | 0.200 | 0.677 | 0.844 | 0.761 |
| Metrica (3) | 0.272 | 0.678 | 0.849 | 0.792 |
| all 10 | 0.222 | 0.677 | 0.846 | 0.771 |
| all 10, before absorbing short runs | 0.194 | 0.665 | 0.830 | 0.771 |

Absorbing short runs raised spellF1 on 7 of the 10 matches; the 3 that fell are IDSSE (by up to
0.008), whose DFL spells are short (`external/README.md`). The cutoffs were picked on the 10
outside matches: spellF1 is flat at 0.221-0.222 for H/A 1-2 s and D 3-5 s.
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
| idsse_J03WMX | 511 | 378 | 206 |
| metrica_1 | 351 | 408 | 224 |
| metrica_3 | 375 | 434 | 215 |

So on the outside matches it now makes about half as many possessions as the truth, yet scores
higher. Absorbing helps Metrica (over-split before) far more than IDSSE (already under).

On BAR-ATH, nothing tuned on it:

| | value |
|---|---|
| public 5 minutes (`run_evaluation.py --bootstrap 1000`), a pipeline check only | macroF1 0.772 (best on the board), spellF1 0.372 (5-95% 0.19-0.57), 12 possessions of mean 20.2 s against 17 of 14.2 s; before absorbing: 0.724, 0.293, 24 of 9.9 s |
| full match, first / second half | 120 / 125 possessions, mean 17.0 / 15.4 s, 36% / 39% dead; before absorbing: 271 / 249, 7.2 / 7.5 s |
| graded frames with no spread lean (a side untracked for the whole window) | 1,277 (0.8%); they keep the previous team |
| graded frames with no player speed (counted as dead) | 0 |

The public window has only 17 possessions, so gaps on its spellF1 under about 0.39 are noise.
A real match has roughly 300 possessions, so 245 is now slightly too few, where 520 was far too
many.

This model is one half of a combined model with `hmm/`, not a replacement for the best single
model.

What it gets wrong:
- **Spell edges are crude.** Absorbing short runs fixed the over-splitting (520 possessions on
  BAR-ATH down to 245), but it merges blindly: a real short possession is lost as easily as a
  false one, and on the outside matches it now makes about half the true count. Next: choose whole
  spells with a length prior (idea 1 below) instead of fixed minimums.
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
