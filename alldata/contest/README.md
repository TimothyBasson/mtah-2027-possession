# alldata/contest: training aimed at brief possessions won under pressure (failure mode 1)

Failure mode 1 of the review (thread "Target final model failure modes"): a player wins the ball
under pressure, keeps it for 2 to 5 s and loses it again, and the model gives the whole
possession to the other team. On the shipped recipe (`pw1e`) that costs 0.149 on folded labels and
0.226 on labels as built. This folder attacks it inside the model: new inputs that describe the
contest, and a loss that weighs these possessions more.

**Result.** The slice's cost drops by about a sixth (0.149 to 0.126 folded, 0.226 to 0.188 as
built, two seeds, down on 29 of 30 matches). Overall the score rises 0.704 to 0.710 folded and
0.626 to 0.641 as built, both with the submission's touch decode. **SkillCorner goes down**
(−0.021 folded, −0.014 as built), so the bar "+0.01 and no provider down" is met on labels as
built only if SkillCorner is set aside. IDSSE and Metrica, whose tracking BAR-ATH uses, go up in
both views. Milder settings do not keep SkillCorner level.

All numbers: held out by provider, BAR-ATH-like balls (version 1), two seeds averaged, adapted
without labels as the shipped nets are (`--pseudo`), `--share skillcorner=0.25`, fixed decode of
`alldata/learned/score.py` plus the touch evidence of `alldata/touch` (weight 1, hold 3 s,
threshold 0.3, as in the submission). Plain all-data nets: Tim's v8 is not an input (retraining
with it needs his held-out classifiers on these balls rebuilt, several hours).

| folded / as built | IDSSE | Metrica | SkillCorner | mean | slice cost (before touches) |
|---|---|---|---|---|---|
| shipped recipe `pw1e` + touches | 0.760 / 0.666 | 0.831 / 0.794 | 0.520 / 0.417 | 0.704 / 0.626 | 0.149 / 0.226 |
| + contest inputs (`pc1e`) | 0.772 / 0.689 | 0.834 / 0.808 | 0.515 / 0.412 | 0.707 / 0.636 | 0.140 / 0.209 |
| + contest inputs, brief x3, focal 1 (`pcr1e`) | 0.791 / 0.706 | 0.839 / 0.812 | 0.500 / 0.404 | **0.710 / 0.641** | **0.126 / 0.188** |

Before adaptation and touches, one seed each (base `w1e` 0.688 / 0.613, slice 0.156 / 0.214):

| | folded / as built | IDSSE / Metrica / SkillCorner folded | slice cost |
|---|---|---|---|
| contest inputs `c1` | 0.690 / 0.617 | 0.739 / 0.821 / 0.512 | 0.153 / 0.208 |
| inputs + brief x3 + focal 1 `cr1` | 0.695 / 0.620 | 0.771 / 0.823 / 0.492 | 0.128 / 0.184 |
| inputs + brief x2 + focal 0.5 `cm1` | 0.691 / 0.621 | 0.747 / 0.827 / 0.498 | 0.143 / 0.192 |
| inputs + focal 1 only `cf1` | 0.683 / 0.613 | 0.724 / 0.815 / 0.509 | 0.159 / 0.227 |
| brief x3 without the inputs `r1` | 0.687 / 0.607 | 0.750 / 0.809 / 0.503 | 0.136 / 0.193 |

Reading (mine): weighting brief possessions makes the net cut finer (predicted possessions per
true one 0.92 to 0.97). IDSSE and Metrica label these brief steals, so they reward it; SkillCorner's
labels are the noisy ones (thread "Clean up noisy training labels") and punish it. The same trade
showed up in the players-only `sp` run. Hard-example mining alone does nothing; the inputs alone
help a little everywhere except SkillCorner (−0.006).

## Review of PR 61 (paired intervals, pseudo-label weighting)

**Paired differences** (`paired.py`, touch decode, mean of provider means, 90% bootstrap over
matches within each provider; share of the 30 matches up, then IDSSE / Metrica / SkillCorner):

| | as built | folded |
|---|---|---|
| inputs `pc1e` − `pw1e` | +0.010 [+0.006, +0.015], 43% up (+0.023 / +0.014 / −0.006) | +0.003 [−0.000, +0.007], 43% (+0.013 / +0.002 / −0.006) |
| reweighted `pcr1e` − `pw1e` | +0.015 [+0.009, +0.021], 37% (+0.040 / +0.018 / −0.013) | +0.006 [+0.002, +0.010], 33% (+0.032 / +0.007 / −0.020) |
| same, pseudo-labels not reweighted `qcr1e` − `pw1e` | +0.015 [+0.008, +0.022], 40% (+0.040 / +0.019 / −0.013) | +0.005 [+0.001, +0.008], 30% (+0.031 / +0.001 / −0.018) |
| seed noise: `pc1s1` − `pc1` | −0.000 [−0.004, +0.003] | +0.004 [+0.002, +0.005] |
| seed noise: `pcr1s1` − `pcr1` | −0.002 [−0.005, +0.001] | +0.002 [−0.001, +0.005] |
| seed noise: `qcr1s1` − `qcr1` | +0.005 [+0.003, +0.007] | +0.002 [−0.002, +0.006] |

The means clear seed noise on labels as built; folded, only the reweighted runs do, and by little.
Most matches are SkillCorner (20 of 30), which goes down, hence the low share up. The variants
were compared on the test folds themselves (five one-seed configurations), so the winner's margin
carries some selection bias.

**Pseudo-label weighting.** `--brief` used to weigh the BAR-ATH (or held-out) pseudo-labelled
matches too; it no longer does (`cfit.py`). Retrained (`qcr1`, `qcr1s1`, `qcrN`, `qcrNs1`): held
out unchanged (0.706 / 0.638 against 0.706 / 0.639, slice 0.124 / 0.190), and BAR-ATH still cuts
175 + 178 possessions (`alldata_contest_q_touch.csv`, sha256 `f0e6a1b5...dc7df41f18`, public
window 0.615). So the over-cutting comes from weighting brief possessions in the labelled
training matches, not from self-training on its own flicker. The inputs-only file (150 + 159) is
on the public pace; if either is used, it is the safer one.

**Focal and `--share`.** Frames not mined (SkillCorner, pseudo) keep a constant 0.5^G while mined
frames fade as the net fits, so their share of the gradient grows over training and
`--share skillcorner=0.25` no longer holds exactly under `--focal`.

**Kink check.** It scores real touches that have a kink within 0.5 s: 52 / 64 / 68% of them
(IDSSE / Metrica / SkillCorner). The 93-96% is the kink's team accuracy on those, not its recall.

## BAR-ATH

All-match nets `models/pcrN.pt`, `pcrNs1.pt` (inputs + brief x3 + focal 1, adapted to BAR-ATH
without labels from `crN` / `crNs1`) and `pcN.pt`, `pcNs1.pt` (inputs only). `python
alldata/contest/run.py` writes `alldata_contest_touch.csv` (sha256 `f9e3c50d...c299d62c`), the
inputs-only file `alldata_contest_inputs_touch.csv` (`72f68c67...7e865d600`); both pass the
validator, a second run is byte-identical. Copies in `/mnt/project-files/alldata/contest/`.

| | possessions (half 1 + 2) | median | under 5 s | public 5 min (17 possessions, a check) |
|---|---|---|---|---|
| submission (with Tim, touches) | 134 + 167 | 7.7 / 5.8 s | 41 / 47% | 0.613 |
| plain all-data + touches | 119 + 149 | 8.0 / 6.6 s | 34 / 43% | 0.650 |
| contest inputs + touches | 150 + 159 | 5.5 / 5.9 s | 47 / 45% | 0.640 |
| contest + brief x3 + focal + touches | 175 + 180 | 4.4 / 4.8 s | 51 / 51% | 0.626 |

The reweighted file cuts BAR-ATH finer than any earlier file (the players-only `sp` run drew 158 /
184, the public pace is about 150 a half). In the public window both contest files find the
20 s Away possession at frame 14,846 and the 3.6 s one at 15,799 that the submission merges
(IoU 0.46 to 0.90-0.92 and 0.08 to 0.58-0.64), and lose elsewhere; nothing was chosen on it.

## The contest inputs (`cfeats.py`)

A smoothed ball rounds every corner. At a one-touch block or a first-time clearance the shown
ball turns 2 to 4 m short of the player who played it, so feats.py's touch columns, which ask who
is nearest the shown ball at the turn, often name the wrong team in exactly the pressured
moments of this failure mode. Away from the corner the shown ball is the true one, so the
straight stretches 0.6 to 0.9 s either side of a turn are run on to where they meet: that point
and time are where the ball was really played (a "kink").

`cfeats.py --check`, real touches of the recorded ball with a kink within 0.5 s, team named right:

| | kink | nearest to the shown ball |
|---|---|---|
| IDSSE, all / opponent within 4 m | 0.983 / 0.960 | 0.885 / 0.800 |
| Metrica | 0.970 / 0.926 | 0.914 / 0.823 |
| SkillCorner | 0.971 / 0.940 | 0.926 / 0.868 |

20 columns at 5 fps: last and next kink (how long ago or ahead, each team's nearest player to the
meeting point, the margin between them, how sharp, how well the lines meet, a signed run of kinks
won by one team), and the ball's present line run backwards and forwards against the players
(who played this ball, who it is travelling to). On BAR-ATH's own ball the columns sit in the
range of the simulated outside matches.

## The loss (`cfit.py`)

`alldata/learned/fit.py` with `--extra` (the columns above), `--brief W` (frames of a possession
under `--brief-len` s that directly follows the other team's possession weigh W, each head on its
own labels) and `--focal G` (each frame's loss scaled by (1 − p of its label)^G; SkillCorner and
pseudo-labelled frames are not mined). Same network, folds, sampling and output layout, so
`score.py`, `breakdown.py` and `alldata/touch/tuse.py` read the runs.

## Files and commands

    python alldata/learned/build.py --no-tim       # the shipped rows (about 1 min)
    python alldata/contest/cfeats.py               # contest columns (about 1 min); --check for the table above
    sh alldata/contest/job.sh c1 idsse --extra     # one fold; each run about 25 min on one CPU
    sh alldata/contest/job.sh cr1 idsse --extra --brief 3 --focal 1
    python alldata/learned/ens.py cr1e cr1 cr1s1
    sh alldata/contest/job.sh pcr1 idsse --extra --brief 3 --focal 1 --pseudo cr1e
    python alldata/contest/slice.py --tags pw1e pcr1e          # overall and slice cost
    python alldata/touch/tuse.py --tag pcr1e --thr 0.3 --w 1 --hold 3 --keep 0   # with the touch decode
    python alldata/contest/run.py --features && sh alldata/contest/bar_nets.sh  # BAR-ATH rows, then the all-match nets
    python alldata/contest/run.py                  # BAR-ATH file: pcrN + pcrNs1 + touches

Nothing outside this folder changes; the shipped nets and files are untouched.
