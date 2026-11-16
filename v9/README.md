# v9/: sequence nets as a clue in the hmm_v8 decoder (experiment, 2026-11-16)

Thread "Rebuild possession model end to end for v9". The decoder change itself ships from PR #47
(`hmm/net_clue.py`, thread "Build v9 with new team-naming inputs"); this folder holds the variants
measured beside it and their scorer. Nothing here labels BAR-ATH.

```bash
sh v9/queue.sh v9/jobs.txt                       # fold nets, three at a time, 75 min a fold on 4 cores
python v9/score.py --nets v9:pf,v9:pfs1,v9:pfs2  # needs external/train_control.py --loso first
```

`v9/train.py` is deep/'s bt network with two switches: `--labels folded` (the target is the label
folded to the control rule, external/control.py) and `--ballin` (the ball, knocked out as the
scorer knocks it out, is an input). `v9/score.py` adds wd x the nets' mean log(p / prior) to the
decoder's emissions next to wc x the control classifier's log-ratio.

Held out by provider, folded spellF1 (external/score_control.py's yardstick; hmm_v8 reproduced at
0.644 / 0.816 / 0.480, mean 0.647). "fair": (wc, wd) picked for each provider on the other two.

| nets in the clue | wc, wd | IDSSE | Metrica | SkillCorner | mean | fair |
|---|---|---|---|---|---|---|
| none (hmm_v8) | 0.2, 0 | 0.644 | 0.816 | 0.480 | 0.647 | |
| `sq`: ball in, folded labels, 1 net | 0.1, 0.2 | 0.662 | 0.825 | 0.492 | 0.660 | 0.657 |
| `pf`: players only, folded labels, 1 net | 0.1, 0.2 | 0.703 | 0.813 | 0.511 | 0.676 | 0.661 |
| `pf` + `sq` at its own weight 0.1 | 0.2, 0.3 | 0.714 | 0.806 | 0.510 | 0.677 | 0.664 |
| `pf`, 3 seeds | 0.2, 0.4 | 0.714 | 0.801 | 0.512 | 0.676 | 0.670 |
| `pf` + #47's `bt`, `b2` (labels as given) | 0.2, 0.3 | 0.724 | 0.809 | 0.503 | 0.678 | 0.673 |
| `pf` 3 seeds + `bt`, `b2` | 0.2, 0.4 | 0.723 | 0.802 | 0.505 | 0.677 | 0.670 |

Reading:
- Every players-only mix lands between 0.672 and 0.678 for wc 0.1 to 0.2 and wd 0.2 to 0.4. More
  seeds steady the fair pick (0.661 to 0.670) and do not lift the best. The family tops out near 0.68.
- Folded-label nets are better on SkillCorner (0.51 against 0.50) and worse on IDSSE (0.70 to 0.71
  against 0.72) than nets taught the labels as given; the mean is the same within noise (0.005).
- The ball as a net input is not worth having: alone the ball-fed net decodes to 0.630 against
  0.644 to 0.648 for players only, and beside a players-only net it adds 0.001 to 0.002. The
  decoder's own ball clues already carry it.
- The nets alone (deep/'s decode, 2 s minimum possession) score 0.648 to 0.650: level with hmm_v8.
  The gain is the two together.
- A provider-adversarial head (`--adv`) is written but was not run: with the family saturating it
  was not the best use of 75 minutes a fold.

Fold nets and probabilities: /mnt/project-files/v9/{sq,sqs1,pf,pfs1,pfs2}/, and `pf` trained on all
30 matches in /mnt/project-files/v9/pf/none/net.pt (loads with deep/net.py load_net).
