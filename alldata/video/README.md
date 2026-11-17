# Our own ball from the BAR-ATH tactical video (2026-11-17 speed run)

**Answer: not adopted. Filling the tracker's ball gaps from a stock video detector does not
improve the submission on any check we are allowed to run, and filling every gap looks slightly
worse.** The submission (`alldata/learned/run.py`, 4979aff0) is unchanged.

## What was built

| step | file | what it does |
|---|---|---|
| frames | `targets.py` | every 12th frame (about 2 Hz) inside the tracker's ball gaps in the graded halves (5,393 frames), plus every 250th frame where the tracker sees the ball, as a check (382 frames) |
| detect | `detect.py` | Ultralytics **YOLO11s, stock COCO weights**, class 32 "sports ball", 1920 px input, conf ≥ 0.05, on the tactical video (whole file, or pieces cut with `ffmpeg -c copy -copyts`). About 0.7 s a frame on 4 CPU cores |
| pitch | `pitch.py` | image → pitch through the provided tactical homography (it maps the ATD's 0-1 pitch to pixels) |
| pick + fill | `ballfill.py` | per frame, the most confident box (conf ≥ 0.3) that lands inside the pitch lines and not within 1 m of a penalty or centre spot (both are painted balls to the detector); picks moving > 30 m/s from both neighbours dropped; picks within 1 s joined by straight lines and to the tracker's ball at the gap's ends; the fill passed through the same 0.7 Hz low-pass as the tracker's ball |
| run | `run.py` | the submission pipeline unchanged on the filled ball (own caches; Tim's v8 input stays on the tracker's ball). `--max-gap 1` rebuilds 4979aff0 byte for byte |
| compare | `check.py`, `compare.py` | detector vs tracker on the check frames; label-free comparison with the submission |

No hand labels, nothing tied to frames of this match: the only choices are the stock model, pitch
geometry, and the conf floor (0.3), which was set from the check frames against the tracker, not
from any labels.

## Detector against the tracker (382 check frames, tracker sees the ball)

| conf floor | ball picked on | picks within 2 m | within 5 m | median error |
|---|---|---|---|---|
| 0.1 | 74% | 67% | 79% | 0.9 m |
| **0.3 (used)** | **57%** | **70%** | **83%** | 0.9 m |
| 0.5 | 45% | 71% | 84% | 0.8 m |

Before the pitch-line and spot filter, most wrong boxes were spare balls by the touchline and the
penalty spots. These are the easy frames (the tracker sees them too); recall inside the gaps is
unknown and likely lower.

## Effect on BAR-ATH

The submission calls 88.6% of the tracker's gap frames dead (13% of seen frames), and gaps over
10 s 87-97%. That figure describes the model, not the match: it calls them dead partly *because*
the ball is missing. The non-circular number is the public window's truth: of its 885 graded
tracker-gap frames, 55.8% are dead (14.7% of seen frames), and the submission calls 58.0% of them
dead. So the gaps are mostly but not overwhelmingly stoppages (a 5-minute sample; the long gaps of
the full match are not in it), and the model's dead calls there are about as frequent as the
truth's.

| file | ball seen (graded) | frames changed | dead share | team spells | public 5 min spellF1 (bootstrap 5-95%) | Smart Tagging phase share vs sub |
|---|---|---|---|---|---|---|
| submission 4979aff0 | 59.3% | – | 45.1% | 301 | 0.613 (0.44-0.77) | 0.870 |
| fill all gaps | 69.4% | 2,923 (1.8%) | 44.4% | 312 | 0.560 (0.40-0.71) | -0.011 [-0.022, -0.002] |
| fill gaps ≤ 10 s | 60.5% | 728 (0.46%) | 45.1% | 302 | 0.560 (0.40-0.71) | -0.001 [-0.003, 0.000] |

The public window (17 possessions) is a pipeline check, not a referee, and was not used to choose
anything; it only rules out a large gain. The tag check is a smoke test. Neither shows a gain, and
the all-gaps fill is worse on the whole-match tag check with a bootstrap band below zero.

## What was not done (honest limits)

- **The null result is about these nets, not about the video ball.** The learned nets were trained
  on balls with tracker-like gaps (alldata/learned feats.bar_like), so a filled BAR-ATH ball is out
  of their training distribution. Retraining with simulated video fills would be the fair test.

- **No held-out score.** The detector cannot be run on the outside matches (no video), so there is
  no spellF1 for this on IDSSE / Metrica / SkillCorner. A simulation (outside ball hidden like
  BAR-ATH's, refilled at the measured 57% / 70% rates) would be the next step; not done tonight.
- Only every 12th gap frame was scanned (CPU time); no tracking between detections, no fine-tuning,
  no ball-in-the-air handling (a lofted ball projects to the wrong place on the ground plane).
- Rule 3: a submission built on this must regenerate from the provided data, so `detect.py` would
  have to run on the full tactical video as part of the pipeline (about 70 min on 4 cores).

## Reproduce

```
python alldata/video/targets.py
python alldata/video/detect.py data/tactical/BAR-ATH_tactical.mp4   # needs ultralytics, opencv
python alldata/video/check.py
python alldata/video/run.py                    # -> evaluation/results/alldata_video_gall.csv
python alldata/video/run.py --max-gap 250      # -> alldata_video_g250.csv
python alldata/video/compare.py alldata_video_gall.csv alldata_video_g250.csv
```
Detections and filled balls are frame-level derived data and stay out of the repository.
