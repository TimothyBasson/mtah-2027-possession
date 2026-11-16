# alldata/briefsafe: the brief-possession gain without the SkillCorner loss

Follows `alldata/contest` (thread "Fix brief possessions via targeted training"). There, contest
inputs plus brief possessions weighted 3x plus hard-example mining (`pcr1e`) raised the held-out
score to 0.710 folded / 0.641 as built, but SkillCorner fell (−0.020 / −0.013). This folder finds out
why and keeps the gain with no provider down. SkillCorner stays at a quarter of training windows
(`--share skillcorner=0.25`) throughout.

## Result

Held out by provider, BAR-ATH-like balls (version 1), the submission's touch decode (threshold 0.3,
weight 1, hold 3 s), plain all-data nets (no Tim input), seeds averaged. Paired differences against
the shipped recipe `pw1e` from `alldata/contest/paired.py`, with a 90% bootstrap over matches.

| | IDSSE | Metrica | SkillCorner | mean, folded / as built |
|---|---|---|---|---|
| shipped recipe `pw1e` | 0.760 / 0.666 | 0.831 / 0.794 | 0.520 / 0.417 | 0.704 / 0.626 |
| brief weighting `pcr1e` (alldata/contest) | 0.791 / 0.706 | 0.839 / 0.812 | 0.500 / 0.404 | 0.710 / 0.641 |
| kink gate on `pcr1e` (`gk10`) | 0.788 / 0.699 | 0.834 / 0.803 | 0.513 / 0.414 | 0.711 / 0.639 |
| ball-backed brief weighting `kbe` | 0.782 / 0.695 | 0.839 / 0.811 | 0.513 / 0.414 | 0.712 / 0.640 |
| **kink gate on `kbe` (`gkb`)** | 0.783 / 0.692 | 0.831 / 0.801 | 0.518 / 0.418 | **0.711 / 0.637** |

| paired vs `pw1e` | as built | folded | providers as built | providers folded | matches up (built / folded) |
|---|---|---|---|---|---|
| `pcr1e` | +0.015 [+0.009, +0.021] | +0.006 [+0.002, +0.010] | +0.040 +0.018 −0.013 | +0.032 +0.007 −0.020 | 37% / 33% |
| `gk10` | +0.013 [+0.009, +0.017] | +0.008 [+0.005, +0.011] | +0.033 +0.010 −0.004 | +0.028 +0.003 −0.008 | 53% / 43% |
| `kbe` | +0.014 [+0.009, +0.019] | +0.008 [+0.004, +0.012] | +0.030 +0.017 −0.004 | +0.023 +0.007 −0.007 | 57% / 43% |
| **`gkb`** | **+0.011 [+0.006, +0.017]** | **+0.007 [+0.004, +0.010]** | +0.027 +0.007 **+0.001** | +0.023 −0.001 **−0.002** | **73% / 53%** |
| seed noise (`kbs1` − `kb`) | +0.001 [−0.002, +0.003] | −0.001 [−0.003, +0.002] | | | |

`gkb` is the only variant with no provider down beyond seed noise (Metrica −0.001 and SkillCorner −0.002
folded are inside ±0.002), and it is up on the most matches. It clears +0.01 on labels as built; folded
it is +0.007.

## After review (PR 62)

- **Do not ship `alldata_briefsafe_touch.csv`.** `bardiff.py` shows that about half of the 4% of
  frames it changes against the submission (49% in half 1, 70% in half 2) are the fine net's dead
  calls replacing those of the Tim-input base. The gate only screens team disagreements, and that
  base has no held-out runs.
- **`--strict`** keeps the base everywhere except kink-backed team swaps (`gate.py --strict`,
  `run.py --strict` -> `alldata_briefsafe_strict_touch.csv`). Held out on `kbe` (`gkbs`), against
  `pw1e`: +0.009 [+0.004, +0.014] as built, +0.006 [+0.004, +0.007] folded; providers +0.016 /
  +0.009 / +0.002 and +0.018 / −0.002 / +0.001; up on 70% / 63% of matches. On top of the
  submission it changes only 0.6% of frames (10 possessions, 80 to 100% kink-backed), and the public
  window is identical (0.613). The Tim-input base already agrees with most of the swaps, so the gate
  adds little to what ships. **The submission stays as it is.**
- **Pseudo windows unweighted** (`kq`, `kqs1`, `kqN`, `kqNs1`; now the default, and `--weigh-pseudo`
  reproduces `kb`). Held out `kqe` +0.012 / +0.008 (SkillCorner −0.005 / −0.007); gated `gkq`
  +0.010 [+0.005, +0.016] / +0.008 [+0.005, +0.011], providers +0.021 / +0.010 / +0.000 and +0.023 /
  +0.002 / −0.002. That is the same as `kb` within seed noise (`kqs1` − `kq` +0.004 / +0.002). On
  BAR-ATH, `kqN` is byte-identical to `kbN`: the pseudo labels there have no brief possession directly
  after the other team's (confidence gaps sit at the turnovers), so nothing was weighted either
  way. The 11% to 18% rise in first-half possessions under 2 s does not come from pseudo weighting.

## Why SkillCorner fell (`scdiag.py`, `support.py`)

- On held-out SkillCorner, `pw1e` predicts 327 possessions against 325 true ones; `pcr1e` predicts 363.
  Precision falls from 0.42 to 0.39 while recall stays almost flat. On IDSSE the shipped net cuts too
  coarsely (298 against 437), so cutting finer helps there. Raising the decode's switch cost 2x or 4x
  hardly changes the counts, because the extra cuts are confident.
- `support.py` looks at the brief possessions `pcr1e` adds over `pw1e`. The label agrees on 0.76, 0.75
  and 0.26 of them (IDSSE / Metrica / SkillCorner). The match's own recorded ball backs the predicted
  team on 0.66, 0.50 and 0.43. On SkillCorner, then, most of the extra cuts are real mistakes, not
  just noisy labels.
- On validation matches (providers seen in training), `pcr1e` also loses SkillCorner (−0.026).

## What fixes it

- **Kink gate** (`gate.py`; no labels, no fitting, the same rule on every provider). Where the
  fine-cut net and the base net name different teams, the fine-cut net is kept only if the last kink
  of the shown ball (`alldata/contest/cfeats.py`: where the ball was really played) was nearer its
  team by at least 0.1 (0.5 m). Otherwise the base stands. It sends back about 55 to 60% of the 2.5%
  of rows where the two disagree. m = 0.1 was the first value tried. 0.3 and 0.5 trade IDSSE for
  nothing, and an "org ray" variant is no better.
- **Ball-backed brief weighting** (`cfit.py --backed 0.1`). Only brief possessions with such a kink
  for the labelled team get the 3x weight (50 to 64% of them). The net then learns to cut where the
  ball shows a contest, rather than wherever a provider's labels cut. Pseudo source `c1e`, one stage
  (`kb`, `kbs1`).

Not helpful: leaving SkillCorner and pseudo windows out of the weighting (`--brief-skip`, `qa`,
`qb`): 0.707 / 0.641 and 0.707 / 0.636 for one seed, no change on SkillCorner. When SkillCorner is
held out, none of its labels are in training anyway. A 50/50 probability blend of `pw1e` and `pcr1e`
gives 0.708 / 0.636: it halves both the loss and the gain.

## BAR-ATH

`python alldata/briefsafe/run.py` puts the gate on top of the actual submission (`tpsN` + `tpsNs1`
with Tim's v8 as an input, keeper fix, touch decode), with `kbN` + `kbNs1` as the fine-cut nets.
It writes `evaluation/results/alldata_briefsafe_touch.csv`, which passes the validator. The
held-out numbers above use the plain base; the with-Tim nets have no held-out runs.
`--base pwN pwNs1 --out alldata_briefsafe_plain_touch.csv` is the exact configuration measured above.

| file | possessions (half 1 / 2) | under 2 s | agrees with submission | public 5 min spellF1 / macroF1 |
|---|---|---|---|---|
| submission `alldata_learned_tim_touch.csv` | 134 / 167 | 11% / 16% | | 0.613 / 0.840 |
| `alldata_briefsafe_touch.csv` (gate on the submission) | 146 / 159 | 18% / 16% | 0.957 | 0.606 / 0.873 |
| `alldata_briefsafe_plain_touch.csv` (gate on pwN) | 141 / 160 | 19% / 17% | 0.952 | 0.617 / 0.883 |

The public window has 17 possessions, so its intervals run from about 0.43 to 0.77; it is a check, not
a referee. Copies of both files, the nets and the held-out runs are in
`/mnt/project-files/alldata/briefsafe/`.

## Commands

    # setup as alldata/contest/README.md, plus the contest thread's runs from /mnt/project-files/alldata/contest/runs/
    sh alldata/briefsafe/queue.sh alldata/briefsafe/q2.txt 4      # kb, kbs1 held out (about 30 min each)
    python alldata/learned/ens.py kbe kb kbs1
    python alldata/briefsafe/gate.py pw1e kbe gkb --m 0.1
    python alldata/contest/paired.py --base pw1e --tags kbe gkb --seeds kb kbs1
    python alldata/briefsafe/scdiag.py --tags pw1e pcr1e         # possession counts, precision / recall per provider
    python alldata/briefsafe/support.py --base pw1e --tag pcr1e  # are the extra brief possessions backed by the ball?
    sh alldata/briefsafe/queue.sh alldata/briefsafe/qN.txt 2      # kbN, kbNs1 for BAR-ATH
    python alldata/briefsafe/run.py                               # BAR-ATH file

Nothing shipped changes: `alldata/learned/run.py` still writes the submission byte for byte
(sha256 4979aff0...).
