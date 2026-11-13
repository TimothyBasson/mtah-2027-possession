# deep/: a learned, players-only possession model

Reads the players' tracking only. The ball is never an input; it is a training target on the
outside matches. Separate from `possession/`, `hmm/` and `seq/`.

```bash
pip install -r requirements-deep.txt
python deep/train.py --fold idsse      # and metrica, skillcorner: one provider held out each
python deep/decode.py --tag v1         # THE score: held-out mean spellF1
python deep/train.py --fold none --steps 2000   # all 30 matches -> copy net.pt to deep/models/v1.pt
python deep/run.py                     # BAR-ATH -> evaluation/results/deep.csv
```

## Result (2026-11-12)

Held out by provider, the same folds and equal-provider mean as `external/heldout.py`:

| held out | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE (7) | 0.541 | 0.877 | 0.949 | 0.895 |
| Metrica (3) | 0.635 | 0.845 | 0.912 | 0.929 |
| SkillCorner (20) | 0.381 | 0.796 | 0.945 | 0.814 |
| **mean** | **0.519** | 0.839 | 0.935 | 0.880 |

For comparison, same protocol: `possession/` on main 0.282; the fitted team model with Viterbi on
`claude/project-thread-3lcb21` 0.435; the classical research branch 0.435 to 0.437.

Each fold trains on the other two providers minus a few validation matches (1 IDSSE, 1 Metrica,
3 SkillCorner where they are in training), and the decode settings are chosen on those validation
matches. The SkillCorner fold therefore learns from only 8 matches, which is the likely reason it
is the weakest; the model that labels BAR-ATH learns from all 30.

BAR-ATH, nothing tuned on it (check only; 17 possessions, so about ±0.2 of noise on spellF1):

| public 5 minutes | spellF1 | macroF1 | alive/dead | team acc | possessions |
|---|---|---|---|---|---|
| `deep` | 0.775 | 0.894 | 0.961 | 0.968 | 15 (truth 17) |
| `possession` (main) | 0.352 | 0.826 | 0.960 | 0.872 | 8 |

Full match: 130 / 150 possessions per half, median 7.8 / 6.6 s, 44% / 48% dead.

## What it is, and where it comes from

- **Players as an unordered set.** Each frame's players go through two self-attention blocks and
  are pooled per team (Set Transformer, Lee et al. 2019; used for exactly this in Kim et al., KDD
  2023, "Ball trajectory inference from multi-agent sports contexts using set transformer and
  hierarchical Bi-LSTM"). No player identity is needed, which suits our 46 swapping track slots.
  Per player: position, velocity, speed, change of speed, team.
- **Time.** Twelve dilated convolutions (0.2 s to 6.4 s steps, twice) and a two-way GRU over the
  5 fps sequence, as the offline task allows (MS-TCN, Farha & Gall 2019; TranSPORTmer, Capellera
  et al. 2024, which labels possession states jointly with trajectories).
- **The ball as a training target.** A second head predicts where the ball is, on live frames
  where it was seen. Kim et al. and TranSPORTmer both find the ball is largely recoverable from
  the players; here that knowledge is forced into the shared features and the head is thrown away.
  A third head marks frames within 0.4 s of any change of label.
- **Symmetry.** The pitch is turned so home defends the left; swapping the teams and mirroring
  the pitch are exact symmetries, used as augmentation and averaged over at prediction.
- **Robustness to tracking.** Position jitter (0.2 m) and hiding up to 15% of players per window.
- **Decode.** A three-state chain with one switching cost, then runs under 1 s absorbed. The
  score is flat across the decode grid (IDSSE 0.527 to 0.541, Metrica 0.630 to 0.636), so the
  network, not the decode, carries the result.

1,500 training steps per fold (2,000 for the final model) of 16 one-minute windows, CPU only,
about 70 minutes per fold on one core.

## What was tried on top (same folds, 1,500 steps, one seed each)

| variant | IDSSE | Metrica | SkillCorner | mean | kept? |
|---|---|---|---|---|---|
| **v1** (above) | 0.541 | 0.635 | 0.381 | **0.519** | yes |
| no ball target (`--ball 0`) | 0.532 | 0.608 | 0.380 | 0.507 | no: the ball target is worth about +0.012, no fold down |
| v1 and no-ball probabilities averaged | 0.539 | 0.623 | 0.391 | 0.518 | no: nothing over v1 |
| short possessions weighted up in the loss (`--spell 0.5`) | 0.552 | 0.653 | 0.371 | 0.525 | no: +0.006, SkillCorner down 0.010 |

By the project's bar (mean up 0.01 or more, no provider down) only the ball target counts, and
with one seed per row even that is suggestive rather than settled.

**Where v1 still loses.** Held out, share of true possessions matched at IoU 0.5, by length:

| | under 2 s | 2 to 5 s | 5 to 10 s | over 10 s | possessions predicted / true |
|---|---|---|---|---|---|
| IDSSE | 8% | 36% | 63% | 87% | 1,813 / 3,061 |
| Metrica | 10% | 44% | 67% | 81% | 714 / 1,041 |
| SkillCorner | 3% | 25% | 47% | 66% | 6,409 / 6,500 |

Long possessions are found; ones under 2 s almost never are, the same wall the hand-built model
hit. Weighting them up in the loss did not move it, so from players alone they may not be there
to find. BAR-ATH's public window has 2 of 17 possessions under 2 s, against 20 to 30% outside.

## Caveats

- One seed per run, so differences under about 0.01 between rows are not established.
- The outside labels are not the organisers' rule (`external/README.md`); 0.519 is not a
  leaderboard prediction.
- `torch` is a new dependency and is not in `uv.lock`.
- A learned model can pick up a provider's tracking quirks. Holding a provider out guards
  against that, and the BAR-ATH public window agrees, but that window is five minutes.
