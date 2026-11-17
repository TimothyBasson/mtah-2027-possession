# alldata/selftrain: self-training and test-time adaptation on the unlabelled target match

Question (2026-11-17): can the all-data model adapt to the match it is predicting, without labels,
beyond the one round of hard pseudo-labels it already has (`fit.py --pseudo`, worth +0.004)?
Simulated fairly: one provider is held out, the model adapts on that provider's matches with no
labels, then is scored on them. Everything is on BAR-ATH-like balls (`score.py --key s`), the
plain all-data model (`--drop-cols tim --share skillcorner=0.25`, SkillCorner a quarter of the
windows), fixed decode. Bar: +0.01 on folded labels, two seeds, no provider down.

**Answer: not adopted.** Only entropy minimisation moves the score, and it is just under the bar
without the touch decode and about half that with it, with Metrica down on the labels as built
both times. The submission is unchanged; no BAR-ATH file was made from it.

## Results (held out by provider, as built / folded; IDSSE, Metrica, SkillCorner)

| recipe | tag(s) | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|---|
| shipped base: one round of hard pseudo-labels from the source ensemble | `pw1e` | 0.652 / 0.747 | 0.781 / 0.824 | 0.416 / 0.518 | 0.616 / 0.696 |
| source only, no adaptation (my rerun, two seeds) | `w1`, `w1s1` | 0.647 / 0.724 | 0.773 / 0.811 | 0.405 / 0.517 | 0.608 / 0.684 |
| iterated self-training: hard pseudo-labels from the adapted two-seed ensemble (seed 0) | `qw1` | 0.644 / 0.753 | 0.775 / 0.820 | 0.418 / 0.513 | 0.612 / 0.695 |
| soft targets from the adapted ensemble (sharpened, confidence-weighted) + strong augmentation consistency (two seeds) | `sa`, `sas1` | 0.648 / 0.749 | 0.770 / 0.824 | 0.419 / 0.519 | 0.612 / 0.697 |
| test-time fine-tune from source nets: mean teacher + strong aug + entropy 0.1 | `ta` | 0.660 / 0.736 | 0.767 / 0.817 | 0.411 / 0.524 | 0.613 / 0.692 |
| same, teacher = adapted ensemble | `tb` | 0.660 / 0.742 | 0.773 / 0.820 | 0.411 / 0.520 | 0.615 / 0.694 |
| test-time fine-tune to soft targets only (two seeds) | `tc`, `tcs1` | 0.658 / 0.743 | 0.779 / 0.822 | 0.413 / 0.523 | 0.617 / 0.696 |
| test-time entropy minimisation only, from source nets (two seeds) | `td`, `tds1` | 0.677 / 0.770 | 0.776 / 0.817 | 0.422 / 0.516 | 0.625 / 0.700 |
| **soft self-training, then entropy 0.3 for 400 steps (two seeds)** | `tf`, `tfs1` | 0.663 / 0.762 | 0.765 / 0.824 | 0.439 / 0.532 | **0.622 / 0.706** |
| same, entropy 0.15 (two seeds) | `tj`, `tjs1` | 0.664 / 0.760 | 0.770 / 0.824 | 0.432 / 0.527 | 0.622 / 0.704 |
| same, entropy 0.3 for 200 steps (two seeds) | `tk`, `tks1` | 0.657 / 0.757 | 0.772 / 0.822 | 0.431 / 0.526 | 0.620 / 0.702 |
| control: `pw1e` probabilities sharpened (temperature 0.5 / 0.3) | | | | | 0.617 / 0.696 |

Two-seed rows are the mean of the per-seed scores. Seed-to-seed spread is up to 0.01.

With the submission's decode (touch evidence, `touchscore.py`, detector t1, weight 1, hold 3 s):

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| `pw1e` + touches (the submission's recipe) | 0.666 / 0.760 | 0.794 / 0.831 | 0.417 / 0.520 | 0.626 / 0.704 |
| `tf`, `tfs1` + touches | 0.674 / 0.770 | 0.771 / 0.826 | 0.433 / 0.530 | 0.626 / 0.709 |
| `tj`, `tjs1` + touches | 0.673 / 0.768 | 0.776 / 0.826 | 0.432 / 0.527 | 0.627 / 0.707 |

## Reading

- Pseudo-labels in any form (hard, iterated, soft, stronger teacher, consistency under
  augmentation, mean teacher) land at the shipped level. The one round already in the model
  takes what there is: the teacher's confident frames are the ones the model already gets right.
- Entropy minimisation is the only thing that moves: +0.006 / +0.010 at best, mostly IDSSE (as
  built and folded) and SkillCorner. Plain sharpening of the same probabilities does nothing, so
  it changes which frames are called, not just how sure the model is. It takes Metrica down
  0.011 to 0.016 on labels as built every time (level folded), and its gain overlaps with the
  touch evidence: on top of the submission's decode it is +0.000 / +0.005, Metrica -0.023.
- So the adaptation that works pushes brief, uncertain stretches to one team, which suits IDSSE's
  and SkillCorner's way of cutting possessions and not Metrica's. Metrica's tracking is BAR-ATH's,
  which is the wrong provider to lose on.
- Not tried: per-match (rather than per-provider) adaptation, adapting the with-Tim nets (their
  inputs need Tim's features rebuilt on simulated balls), the BAR-ATH public window (not made,
  since nothing cleared the bar).

## Running

```bash
python alldata/learned/build.py --versions 3 --no-tim       # feature cache
sh alldata/learned/job.sh w1 idsse --drop-cols tim --share skillcorner=0.25        # source net (per fold, seed)
sh alldata/selftrain/job.sh sa idsse --teacher pw1e --strong                       # soft self-training
sh alldata/selftrain/job.sh tf idsse --init sa --teacher pw1e --lam 0 --ent 0.3 --lr 3e-4 --const-lr --steps 400
python alldata/learned/score.py --tags pw1e tf --key s
python alldata/selftrain/touchscore.py pw1e tf             # with the submission's touch decode
```

`pw1e` (the teacher) is the saved held-out ensemble in `/mnt/project-files/alldata/learned/runs/`.
A source run takes about 17 minutes a fold on one CPU; a 400-step fine-tune about 5. Saved
held-out probabilities of every run are in `/mnt/project-files/alldata/selftrain/runs/`.
