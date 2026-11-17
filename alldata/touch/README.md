# alldata/touch: brief touches read off the smoothed ball (failure mode 2, knock-on)

Failure mode 2 of the review (thread "Target final model failure modes"): a missed brief touch
merges the possessions either side, so one miss costs up to three. This folder attacks it from
the ball: detect the touches themselves from the smoothed ball's change of velocity beside a
player, then use them in the decode and as an input.

**Superseded as the submission (2026-11-17).** Gambtt chose hmm_v9 (`python -m hmm.decode`, built by `submission/make_submission.py`) as the submitted file; see submission/TECHNICAL_DOC.md. Everything below still holds and the file is still built by `python alldata/learned/run.py` (or `make_submission.py --method alldata`); it is no longer the one uploaded.

**Status.** The submission from 2026-11-16 18:34Z to 2026-11-17 (Gambtt: "yes make it the sub"): `python alldata/learned/run.py` writes `alldata_learned_tim_touch.csv`.

**Result.** A learned touch detector works on the smoothed ball and on an unseen provider, and
as decode evidence it adds **+0.010 / +0.008** (as built / folded) without retraining anything;
read by the net as well, +0.015 / +0.012. No provider goes down. Most merges stay: which
touches a provider labels as a possession does not transfer, and forcing them costs more than it
gains.

All numbers: held out by provider, BAR-ATH-like balls (version 1, parabolic; version 4, the
0.7 Hz low-pass, where stated), mean of IDSSE / Metrica / SkillCorner, fixed decode of
`alldata/learned/score.py`. Base is `pw1e`, the plain all-data model (0.616 / 0.696); held-out
predictions of the with-Tim submission model were never saved, so it could not be scored here.

| | IDSSE | Metrica | SkillCorner | mean as built | mean folded |
|---|---|---|---|---|---|
| base `pw1e` | 0.652 / 0.747 | 0.781 / 0.824 | 0.416 / 0.518 | 0.616 | 0.696 |
| + touch evidence in the decode (`tuse.py`, w 1, hold 3 s) | 0.666 / 0.760 | 0.794 / 0.831 | 0.417 / 0.520 | **0.626** | **0.704** |
| touch columns as net input (`xpw1e`, two seeds) | 0.667 / 0.756 | 0.788 / 0.824 | 0.414 / 0.520 | 0.623 | 0.700 |
| input + evidence | 0.684 / 0.769 | 0.797 / 0.835 | 0.413 / 0.518 | **0.631** | **0.708** |
| input + evidence, low-pass ball (base there 0.632 / 0.707) | 0.705 / 0.787 | 0.807 / 0.836 | 0.405 / 0.510 | 0.639 | 0.711 |
| ceiling: touches of the recorded ball as evidence (w 4, hold 2 s) | 0.746 / 0.804 | 0.821 / 0.850 | 0.416 / 0.514 | 0.661 | 0.723 |

Evidence alone is up on 22 / 23 of the 30 matches (as built / folded). The evidence setting
(weight 1, hold 3 s) was picked on the validation matches of each fold (`--kind val`), where
heavier settings lose; the best setting read off the test table would give 0.628 / 0.707.

Merge slice (true possessions under 3 s directly between two possessions of the other team;
IDSSE / Metrica / SkillCorner): brief touches matched at IoU 0.5 go 19 / 45 / 7% to 25 / 50 / 7%
with evidence, neighbours 42 / 60 / 32% to 43 / 63 / 32%, triples merged into one possession
42 / 31 / 60% to 39 / 30 / 60%. The gain is real, the merges mostly stay.

## The detector (`tfeats.py`, `tfit.py`)

Target: the touches of the ball as recorded (`hmm.ball.detect_touches`, the detector behind the
folded labels), with the nearest team, +-2 frames. Input at 25 fps: the BAR-ATH-like ball's
velocity and its change over 0.16 / 0.4 / 0.8 s, and each team's three nearest players relative
to the ball (offset, distance, relative velocity), on ball versions 1-5. A dilated conv net,
three classes, team swap and mirrors as augmentation and averaged at prediction. One provider
held out, 3,000 steps, about 25 minutes a fold.

Unseen provider, peaks over 0.3, same team within 8 frames: recall 0.69 / 0.90 / 0.87,
precision 0.90 / 0.68 / 0.48. The hand-set detector run on the same smoothed ball adds nothing
(-0.002 to -0.02 as evidence), so the learned shape is what makes it work. Brief sandwiched
possessions show an own-team touch in the recorded ball 75 / 75 / 32% of the time, and in the
detector 58 / 75 / 37%.

On BAR-ATH's own ball (`tbar.py`, the all-match detector `models/t1.pt`) it fires 31 times a
minute of seen ball (simulated: 24 to 43), median score 0.88 (simulated 0.89 to 0.94). In the
public window its touches carry the labelled team almost everywhere; of the review's two brief
Home touches it sees the 0.8 s one (score 0.97) and not the 1.3 s one.

## Using the touches

- **Evidence in the decode (`tuse.py`).** After each detected touch, weight x score is added to
  that team's log-probability until the next touch or `hold` seconds. Adopted (see the table).
- **Input columns (`textra.py`, `alldata/learned/fit.py --extra`).** Touch probability now, last
  and next touch per team fading over a second, who touched last / next within 5 s. Each match
  takes the detector of the fold that held its provider out. +0.007 / +0.004 alone, two seeds.
  No BAR-ATH file with these nets was made (another hour of training for +0.005).
- **Forced possessions (`tinsert.py`), not adopted.** Every touch by K inside a possession given
  to O, with O touching again within a few seconds, becomes K's. It cuts merges hard (Metrica
  triples merged 31% to 18%) but scores -0.01 to -0.02 as built and -0.04 to -0.07 folded: about
  half of these touches are not labelled possessions (a deflection, a block, a touch the provider
  does not count).
- **A learned gate on those candidates (`tgate.py`), not adopted.** A boosted-tree classifier on
  the touch scores, the net's own probability, timing, distances and position in the possession.
  On an unseen provider its AUC is 0.50 to 0.57 and every threshold loses. Forcing only the
  candidates the truth agrees with would give +0.030 / +0.030, so that is what is left on the
  table; reading (mine): which brief touches a provider calls possessions is its own convention,
  as the turnover detector found for players-only.

## Touches at ball gaps (review of PR 55)

The detector is never taught on frames without a ball, and a gap that ends makes a kink in the
interpolated path. Of the 1,929 touches it finds on BAR-ATH, 0.5% sit on an unseen frame and
4.2% within 10 frames after a gap ends; on the simulated held-out matches 1.6% and 2.6%. Leaving
those touches out held out scores 0.624 / 0.703 against 0.626 / 0.704 (worse on 25 and 26 of 30
matches), so they are kept.

## Files and commands

    python alldata/touch/tfeats.py            # 25 fps rows and targets (about 2 min)
    python alldata/touch/tfit.py --fold idsse --tag t1   # and metrica, skillcorner; --fold none for BAR-ATH
    python alldata/touch/oracle.py            # ceiling: recorded-ball touches as evidence
    python alldata/touch/tuse.py --tag pw1e [--kind val] [--v 4]
    python alldata/touch/tuse.py --tag pw1e --kind val --thr 0.3 --w 1 2 3 --hold 2 3 --keep 0   # the sweep that picked w 1, hold 3 s
    python alldata/touch/tinsert.py; python alldata/touch/tgate.py
    python alldata/touch/textra.py --touch t1 && sh alldata/learned/folds.sh xpw1 --drop-cols tim --share skillcorner=0.25 --extra data/cache/alldata/touch/extra/t1 --pseudo w1e
    python alldata/learned/run.py             # BAR-ATH: submission nets (tpsN, tpsNs1) + touch evidence (run.py here)
    python alldata/learned/run.py --no-touch --out alldata_learned_tim.csv   # the earlier file byte for byte (75139e82...)

BAR-ATH files (copies in `/mnt/project-files/alldata/touch/`): `alldata_learned_tim_touch.csv`
(sha256 e9f09d6a...), `alldata_learned_touch.csv` (plain all-data nets + evidence, 7ca261ee...).
Against the submission 1.0% of frames differ and the match has 307 possessions instead of 281.
Public window: 0.613, the same as the submission (17 possessions, a check only). Nothing in
`alldata/learned/` changes except the optional `--extra` flag of `fit.py`; the shipped nets and
files are untouched.
