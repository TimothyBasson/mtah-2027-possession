# alldata/robust: is the all-data model robust to BAR-ATH's smoothed ball?

Builds on `alldata/learned/` (PR 43). The question: the learned all-data model scored 0.635 /
0.707 held out by provider on balls as recorded (labels as built / folded), 0.493 / 0.601 when a
raw-ball model met a BAR-ATH-like ball, and 0.602 / 0.692 once trained on BAR-ATH-like balls. Can
a better simulation of BAR-ATH's ball, or training on a mix of smoothing levels, win some of
that back?

**Short answer: no training change clears the bar, and the model does not need one.** What
changed is the measurement. BAR-ATH's ball is not a windowed average; it is a sharp low-pass at
0.7 Hz. Simulated that way, the model that is already shipped scores about **0.61 / 0.70**, not
0.60 / 0.69, and it holds that score over a wide range of filter settings. The cost of the
smoothing against a raw ball is about 0.02 / 0.01, not 0.03 to 0.10.

## What BAR-ATH's ball is (label-free)

`spectrum.py` and `fitfilter.py` compare the power spectrum of the ball's movement, BAR-ATH
against raw outside balls and against raw balls passed through candidate filters (log10 power,
m2/s2 per Hz):

| | 0.2 Hz | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 2.0 | distance |
|---|---|---|---|---|---|---|---|---|---|
| BAR-ATH | 1.84 | 0.87 | -0.50 | -2.11 | -3.53 | -4.54 | -5.30 | -5.86 | |
| raw Metrica / IDSSE ball | 1.7 | 1.0 | 0.8 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | |
| parabolic window, half-width 16 (PR 43's simulation) | 1.78 | 0.49 | -0.26 | -1.30 | -2.46 | -1.88 | -2.02 | -3.74 | 1.85 |
| Gaussian, sigma 10 frames | 1.74 | 0.11 | -0.86 | -1.97 | -3.19 | -4.53 | -5.54 | -5.94 | 0.35 |
| **zero-lag Butterworth, order 5, 0.7 Hz** | 1.83 | 0.86 | -0.24 | -2.07 | -3.76 | -4.91 | -5.46 | -5.86 | **0.18** |

BAR-ATH keeps everything up to 0.6 Hz and nothing above 1.5 Hz (the floor near -5.8 is its
coordinates' rounding to 1e-5 of the pitch). The parabolic window does the opposite on both
sides: it removes more of the slow movement and leaves 10 to 100 times more of the fast. The
low-pass also matches the model's own inputs better (`featstats.py`; ball seen and moving):

| | speed, 90th pct | velocity change, 90th / 99th pct | frames over the touch threshold |
|---|---|---|---|
| BAR-ATH, halves 1 / 2 | 15.4 / 15.2 m/s | 3.4 / 3.6, 8.1 / 13.6 | 29% / 28% |
| outside, parabolic window (6 matches) | 13.3 to 14.7 | 2.8 to 3.2, 4.9 to 5.6 | 16% to 25% |
| outside, low-pass (6 matches) | 14.3 to 15.6 | 3.6 to 4.2, 6.1 to 10.0 | 21% to 32% |

`feats.lowpass` is the filter; `feats.bar_like(style="lp")` draws its order from 4 to 6 and its
cut-off from 0.62 to 0.80 Hz; `style="mix"` draws widely (either filter, 0.45 to 1.4 Hz or
half-width 8 to 24, and a lead or lag of up to 0.2 s). `build.py --versions 10` writes versions
1 to 3 (parabolic), 4 to 6 (low-pass), 7 to 10 (mix); `fit.py --vers` picks which to train on.

## Training on it: held out by provider, one seed

spellF1, labels as built / folded, fixed decode, each head on its own labelling. Columns are the
ball the held-out matches are shown with. Players-only bt: 0.545 / 0.641.

| trained on | low-pass (closest to BAR-ATH) | parabolic window | mix |
|---|---|---|---|
| parabolic window (`a0`, PR 43's recipe) | 0.614 / 0.696 | 0.594 / 0.683 | 0.604 / 0.690 |
| low-pass (`lp0`) | 0.608 / 0.691 | 0.562 / 0.670 | 0.589 / 0.678 |
| low-pass and mix (`mx0`) | 0.616 / 0.690 | 0.587 / 0.678 | 0.602 / 0.686 |
| that, with the half's smoothness as an input (`mxs0`) | 0.612 / 0.692 | 0.587 / 0.677 | 0.602 / 0.683 |

On the low-pass ball all four are within 0.008 on the labels as built and 0.006 folded, which is
inside seed noise (up to 0.01), and no provider moves the same way under both labellings
(`mx0` against `a0`: IDSSE -0.015 / -0.021, Metrica +0.012 / +0.003, SkillCorner +0.009 /
+0.002). Nothing clears the bar of +0.01 with no provider down, so the second seed was not run
and the shipped nets (`psN`, `psNs1`) stay.

## How much the score depends on the simulation (`stress.py`)

The fold nets re-run with the held-out matches' ball under twelve fixed simulations (same hidden
stretches in each):

| ball shown | `a0` (shipped recipe) | `mx0` (mix-trained) |
|---|---|---|
| low-pass order 5 at 0.70 Hz (BAR-ATH fit) | 0.610 / 0.700 | 0.615 / 0.695 |
| at 0.55 Hz (smoother) | 0.599 / 0.693 | 0.605 / 0.688 |
| at 0.45 Hz (much smoother) | 0.583 / 0.681 | 0.585 / 0.675 |
| at 0.90 Hz (sharper) | 0.620 / 0.706 | 0.620 / 0.703 |
| at 1.20 Hz (much sharper) | 0.619 / 0.704 | 0.624 / 0.705 |
| order 2 at 0.70 Hz (gentler slope) | 0.615 / 0.702 | 0.613 / 0.695 |
| 0.70 Hz, 0.32 s late | 0.568 / 0.669 | 0.569 / 0.666 |
| 0.70 Hz, 0.32 s early | 0.578 / 0.665 | 0.570 / 0.658 |
| 0.70 Hz, ball seen through every stoppage | 0.605 / 0.689 | 0.612 / 0.689 |
| parabolic window 16 | 0.594 / 0.686 | 0.586 / 0.676 |
| parabolic window 8 | 0.623 / 0.710 | 0.622 / 0.700 |
| parabolic window 24 | 0.542 / 0.648 | 0.538 / 0.646 |

The shipped recipe is already flat from 0.55 to 1.2 Hz (0.60 to 0.62 / 0.69 to 0.71), and the
mix-trained net is no flatter. The one thing that hurts either is a ball a third of a second
out of step with the players (-0.04), and training with leads and lags of up to 0.2 s did not
remove that; PR 43's `lagcheck.py` puts BAR-ATH's ball within 0.1 s of its players.

## What to take from it

- Quote the all-data model at about 0.61 / 0.70 for BAR-ATH-like balls (one seed, base net
  without the label-free adaptation, which added +0.004 in PR 43).
- A raw ball is worth only about 0.02 / 0.01 more (0.635 / 0.707), so the smoothing is no longer
  the biggest lever; what is left sits where it did before: how providers cut possessions, and
  SkillCorner.
- These are still outside matches with a simulated ball, not a score on BAR-ATH.

## Reproduce

```bash
python alldata/robust/spectrum.py && python alldata/robust/fitfilter.py && python alldata/robust/featstats.py
python alldata/learned/build.py --versions 10 --no-tim   # about 5 minutes
sh alldata/robust/queue.sh alldata/robust/jobs1.txt      # 12 runs, about 65 minutes on 4 CPUs
for k in s l m; do python alldata/learned/score.py --tags deep:bt a0 lp0 mx0 mxs0 --key $k; done
python alldata/robust/stress.py --build --tags a0 mx0    # about 40 minutes
```

`fit.py` without `--vers` trains on versions 1 to 3 as before, and `run.py` still writes the
same BAR-ATH file (sha256 `0c1ce4a7...c58b935`, checked after these changes). Needs scipy.
