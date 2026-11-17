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

## Not in this folder

Other teams' published BAR-ATH files are not our algorithm's output, and some of them copy or
are tuned on the public ground truth. They are not used to justify or change anything in our
submission.
