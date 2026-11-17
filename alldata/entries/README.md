# What other public entries do, and what we can take

Survey of the public GitHub repositories for More than a Hack 2027 Phase 1, made on 2026-11-17.
Nothing here changes the submission on main.

## The survey

- About 450 public repositories match the challenge (searches on the challenge name, `mtah2027`,
  `spellF1`, `BAR-ATH`). 42 of them commit a BAR-ATH label file (445 distinct files, many
  versions per team).
- Almost every serious entry has the same skeleton as ours: per-frame evidence or a learned
  frame model, trained on outside matches made to look like BAR-ATH's tracking, then a Viterbi
  or semi-Markov decode with minimum durations. Outside data used: IDSSE/DFL (most), Metrica
  sample games, SkillCorner, and PFF FC World Cup 2022 (64 matches, free after registration).
- Ideas we had already tried, under another name: team-swap and mirror averaging (we average
  four symmetric views), explicit-duration decoding, restart-team states, second-stage stacking
  over frame probabilities (our whole-match context pass), MBR / expected-spellF1 decoding,
  per-player attention or pointer nets (our holder network), provider flags or provider heads
  for SkillCorner, touch layers, label harmonising to controlled possession (our folded labels),
  Smart Tagging and broadcast cues.
- Ideas we have not tried:
  - **PFF World Cup 2022 as a training source.** Used by False Five, Tiki-Data, Opus Operandi
    and FlickNet. Where it was measured it is small or mixed: FlickNet +0.008 on SkillCorner and
    level or down on Metrica; Opus Operandi kept 20 PFF matches at half weight for one bench and
    rejected 56 matches. Not worth a registration-gated, multi-hour retrain today.
  - **Own ball detection from the video** (Tiki-Data: YOLO detectors on both feeds, fused with
    the tracker's ball). Needs the 14 GB video and cannot be scored held out.
- Things others report as not working match our record (boundary snap, edge shifts, seed
  bagging, weighting rows near possession changes, longer minimum stoppages).

## Boundary shift (team-switch delay), tested

`shift.py` moves every decoded hand-over (or every live/dead edge) of the plain all-data model's
held-out runs (pw1e, BAR-ATH-like ball) by -12 to +12 frames. Baseline 0.616 / 0.696; the best
shift is level (+0.000 / +0.0003), every larger shift loses. Our edges are already centred.
Output: `/mnt/project-files/entries/shift_pw1e.txt`.

## The published files on BAR-ATH

Public window (17 possessions, not a ranking): our submission 0.613 is 13th of 42 team-best
files; the plain all-data file 0.653. The files above 0.75 include a copy of the public ground
truth and files tuned on it. Full board: `/mnt/project-files/entries/published_files_public_window.csv`.

Over the whole match the strongest independent files (SAM Analytics, Tiki-Data, False Five,
HSA Connection, Around the World) agree with our submission on 81-92% of graded frames and
call 216-326 possessions (ours 301). Where all five agree with each other, our file differs on
1.8% of graded frames, almost all in runs under 3 s (edges), except one:

**Frames 168908-170167 (second half, 50 s).** Our submission says Home throughout; 441 of 445
published files and our own plain all-data model say dead ball (the players-only sw file and
Tim's v8 say Home, like the submission). The tracking agrees: the ball is lost near the touchline and the
players walk at about 0.5 m/s for 40 s. The Tim-input model holds Home through it, so a 12.7 s
and a 2.4 s Home possession are merged into one 65.5 s possession that matches neither.

## Long-stoppage guard

`stopfix.py`: inside every dead run of at least 30 s in the plain all-data file, a possession
run of the submission lasting at least 10 s becomes dead. On BAR-ATH it changes exactly that one
stretch (1,260 frames) and leaves the 12.7 s and 2.4 s Home possessions as the plain model and
the other entries have them. The public window is untouched (0.613).

    python alldata/entries/stopfix.py --sub evaluation/results/alldata_learned_tim_touch.csv \
        --plain evaluation/results/alldata_learned.csv --out evaluation/results/alldata_learned_tim_touch_stopguard.csv

Result: `/mnt/project-files/entries/alldata_learned_tim_touch_stopguard.csv` (sha256 635ad7e0425f3177...,
passes the validator). It cannot be scored held out: the with-Tim model's held-out predictions
were never saved, and a 50 s miss of this kind is rare in the outside matches. If the other
entries are right about that stretch, it recovers two possessions of about 300, roughly +0.005
to +0.008 spellF1. With a 5 s threshold it would also touch three short stretches where the
outside files split, so 10 s is the safer setting.
