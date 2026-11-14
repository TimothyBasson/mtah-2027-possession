# deep/spell/: training on whole possessions instead of frames (experiment, not adopted)

Question: spellF1 counts whole possessions with the right start and end, but `deep/` is trained
frame by frame and decoded afterwards. Does a training target shaped like the score do better?

Answer, 2026-11-14: no, not by enough. Every variant lands at 0.623 to 0.630 against 0.627 on
control-rule labels, held out by provider. IDSSE and Metrica gain 0.01 to 0.02, SkillCorner
loses 0.02 to 0.03, in every variant. Nothing here is used by `deep/run.py`.

## What was built

- `crf.py`: a chain over H / A / D in which a run must last a minimum time (2 s for a possession,
  1 s for a stoppage), with a learned cost per kind of change. It is both the loss and the
  decode: loss = log Z(all paths) - log Z(paths that agree with the labels). Checked against
  brute force. Options: a margin charged per wrong frame, weighted so that each possession counts
  the same (`--margin`, `--spell`), and training on the score itself, by drawing paths from the
  chain and scoring each with spellF1 (`--risk`).
- `extract.py`, `head.py`: the v2 fold nets frozen, and only what reads their 128-d state
  trained (a linear layer, or a two-way GRU). Minutes per variant.
- `finetune.py`: every weight of the fold net trained 500 more steps through the chain, and the
  same budget spent on the per-frame loss as the control.
- `baseline.py`: the fold nets as they are under fixed decodes, on both label views.

```bash
for f in idsse metrica skillcorner; do python deep/spell/extract.py --fold $f; done
python deep/spell/baseline.py
sh deep/spell/queue.sh "crf_lin_m2s|--loss crf --margin 2 --spell 0.5"
SCRIPT=finetune sh deep/spell/queue.sh "ft_crf|--loss crf --margin 2" "ft_ce|--loss ce"
python deep/spell/table.py
```

## Results

Held out by provider, spellF1, one seed, trained on control-rule labels. No decode setting is
picked on the held-out provider; the chain's evidence scale is picked on each fold's validation
matches. IDSSE / Metrica / SkillCorner | mean.

| | labels as given | control-rule labels |
|---|---|---|
| v2, shipped decode (cost 4, 1 s) | 0.549 0.635 0.398 \| 0.527 | 0.675 0.720 0.451 \| 0.615 |
| **v2, runs under 2 s absorbed (the bar)** | 0.534 0.624 0.419 \| 0.526 | 0.674 0.719 0.486 \| **0.627** |
| v2, the chain as a decode only, flat cost 8 | 0.544 0.627 0.413 \| 0.528 | 0.675 0.722 0.472 \| 0.623 |
| frozen net, new linear head, per-frame loss | 0.549 0.627 0.404 \| 0.527 | 0.673 0.718 0.461 \| 0.617 |
| frozen net, GRU head, per-frame loss | 0.535 0.625 0.419 \| 0.526 | 0.669 0.722 0.485 \| 0.625 |
| frozen net, linear head, sequence loss | 0.562 0.650 0.399 \| 0.537 | 0.680 0.738 0.450 \| 0.623 |
| frozen net, GRU head, sequence loss | 0.568 0.659 0.404 \| 0.544 | 0.678 0.738 0.459 \| 0.625 |
| ... linear, margin 2 | 0.552 0.635 0.406 \| 0.531 | 0.683 0.731 0.462 \| 0.625 |
| ... linear, margin 2 per possession | 0.559 0.645 0.408 \| 0.538 | 0.688 0.738 0.465 \| 0.630 |
| ... then 300 steps on sampled spellF1 | 0.563 0.638 0.408 \| 0.536 | 0.686 0.738 0.464 \| 0.630 |
| whole net, 500 steps, sequence loss | 0.546 0.636 0.413 \| 0.531 | 0.682 0.739 0.470 \| 0.630 |
| whole net, 500 steps, per-frame loss (control) | 0.512 0.619 0.426 \| 0.519 | 0.658 0.725 0.501 \| 0.628 |

(The per-frame rows on control-rule labels use the 1 s shipped decode for the heads and the 2 s
absorb for the whole-net control.)

## What it says

- The sequence loss is a real but small and uneven effect: +0.01 on IDSSE, +0.02 on Metrica,
  -0.02 to -0.03 on SkillCorner against the 2 s absorb decode. The mean moves by 0.003 at most,
  under the 0.004 seed noise, and one provider is down, so it does not clear the bar.
- SkillCorner wants more smoothing than the two providers its fold trains on (the whole-net
  chain scores 0.486 there at evidence scale 1 and 0.470 at the scale its validation matches
  pick). A trained decode carries the training providers' rhythm to the unseen one, which is the
  provider-shift problem again rather than a problem with the objective.
- On labels as given the sequence heads are up on all three providers (0.544 against 0.527), but
  those labels are not the organisers' rule.
- A chain fitted by likelihood alone over-smooths badly (0.53 on Metrica until its evidence is
  scaled up 5 to 8 times): frames repeat each other, so the calibrated chain trusts them little
  and its best path joins possessions. The margin fixes that in training.
- Training on the score itself adds nothing over the margin (0.630 both).
- Together with the team-naming thread's finding (most of the loss is turnovers never drawn),
  this says the limit is what the features know about turnovers, not how the output is trained
  or decoded.
