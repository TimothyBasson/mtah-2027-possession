# deep/: a learned, players-only possession model

Reads the players' tracking only. The ball is never an input; it is a training target on the
outside matches. Separate from `possession/`, `hmm/` and `seq/`.

```bash
pip install -r requirements-deep.txt
python deep/train.py --fold idsse      # and metrica, skillcorner: one provider held out each
python deep/decode.py --tag v2         # THE score: held-out mean spellF1
python deep/train.py --fold none --steps 2000   # all 30 matches -> copy net.pt to deep/models/v2.pt
python deep/run.py                     # BAR-ATH -> evaluation/results/deep.csv (the ball-taught model, bt)
python deep/explain.py --match all     # each stage's output, for viz/combined/make_combined.py --deep
```

## sw (2026-11-16): trust SkillCorner's labels less, 0.558

Asked: are the outside labels noisy, and does the model learn better from cleaner ones? Checked
against each provider's own ball (`python deep/clean/diag.py`; the ball judges labels here,
nothing else):

| | IDSSE | Metrica | SkillCorner |
|---|---|---|---|
| H<->A changes with a touch by the gaining team within 0.25 s | 75% | 62% | 19% |
| the same within 1 s | 81% | 70% | 45% |
| label change minus that touch, quartiles (s) | -0.04, +0.04 | -0.28, -0.04 | -0.88, +0.32 |
| live frames whose last touch was by the other team | 4.0% | 1.7% | 10.2% |
| label names the other team while one team is clearly on the ball for 1 s or more | 2.8% | 1.0% | 6.7% |

SkillCorner's labels (broadcast tracking, possession held through passes and loose balls,
`external/build.py`) disagree with its ball two to four times as often as the others', and moving
its changes onto touches removes little of that (10.2% to 9.5%): it is not a timing offset. It is
20 of the 30 matches and, with providers drawn equally, half the training windows of the IDSSE
and Metrica folds.

Two remedies, each on the bt recipe, scored by `deep/fixed.py` on labels nobody changed (as built
at the shipped decode; folded to the controlled-possession rule at 2 s / 3 s):

- **`--share skillcorner=0.25`** (`lcsw`): SkillCorner is drawn for a quarter of the training
  windows instead of half. In the all-30 model a quarter instead of a third.
- **`--clean retime+lastmask+mask`** (`lccl`, `deep/clean/labels.py`): on training matches only,
  every H<->A change moves to the gaining team's nearest touch within 1.5 s, and live frames the
  ball contradicts (the last touch within 2 s was the other team's, or the other team is clearly
  on the ball for 0.6 s or more) are left out of the loss. Applied to every training provider:
  it leaves out 4% / 2% / 12% of live frames.

spellF1, mean of two seeds unless said (IDSSE / Metrica / SkillCorner / provider mean):

| | labels as built | folded labels |
|---|---|---|
| bt, bts1 | 0.582 / 0.651 / 0.403 / **0.545** | 0.692 / 0.735 / 0.498 / **0.641** |
| SkillCorner at a quarter (`lcsw`, `lcsws1`) | 0.608 / 0.666 / 0.403 / **0.558** | 0.704 / 0.745 / 0.498 / **0.649** |
| SkillCorner at a tenth (`lcsw10`, one seed) | 0.598 / 0.671 / n.a. | 0.689 / 0.742 / n.a. |
| cleaned labels (`lccl`; one seed, SkillCorner two) | 0.600 / 0.665 / 0.410 / **0.558** | 0.700 / 0.738 / 0.503 / **0.647** |
| both (`lccs`, `lccss1`; SkillCorner fold is `lccl`'s) | 0.602 / 0.669 / 0.410 / **0.560** | 0.700 / 0.745 / 0.503 / **0.649** |

- **Down-weighting SkillCorner is a real gain on the other two providers**: both seeds are up on
  all 10 IDSSE and Metrica matches as built (+0.021 and +0.023, standard error 0.004), 8 of 10
  folded. bt's two seeds differ by 0.001. The SkillCorner fold trains on no SkillCorner, so its
  score is bt's by construction: no provider is down, one cannot move.
- **Cleaning gains about as much, and the two do not add.** Together they score what either
  scores alone (0.560 against 0.558), within seed noise. Cleaning is the only one that moves the
  SkillCorner fold: +0.007 as built on both seeds (15 and 14 of 20 matches up), from cleaner
  IDSSE and Metrica labels. Reading: both work by taking weight off frames where SkillCorner's
  label and the play disagree; cleaning does it frame by frame, the share does it wholesale.
- **A tenth is no better than a quarter**: the folded gain goes on IDSSE. SkillCorner's matches
  still teach something.
- **Bar.** +0.013 to +0.015 as built with no provider down clears the +0.01 bar; on the folded
  labels the gain is +0.007, below it. Team accuracy and alive/dead do not move (0.898 to 0.899,
  0.939 to 0.938): as with bt, the gain is in where possessions start and end.
- **Shipped as an option, not the default**: `deep/models/sw.pt` is the all-30 model on the
  simpler recipe (`--fold none --steps 2000 --ptr 0.5 --share skillcorner=0.25`, seed 0, one
  thread). `python deep/run.py --tag sw --out deep_sw.csv`. Cleaning needs the ball only to build
  training labels, which the ball rule allows, but it adds a touch detector (`hmm.ball.detect_touches`,
  as `external/control.py` uses for the folded yardstick; no possession model is involved) for no
  further gain, so it stays a flag.
- **BAR-ATH public 5 minutes, check only:** sw 0.745 with 12 possessions, bt 0.745 with 12
  (truth 17); macroF1 0.900 against 0.904. Over the match the two agree on 97.3% of graded frames
  (272 against 275 possessions, median 7.1 s both, 11% against 13% under 2 s, 46% against 45% dead).
  The window cannot tell them apart; the 30 outside matches are the evidence.

Not tried: folding SkillCorner's labels only; a noise-robust loss; the share in the SkillCorner
fold's sense (weighting IDSSE against Metrica). Folding every provider's training labels was v4
(not kept, IDSSE and Metrica down).

```bash
python deep/clean/diag.py                    # the table above
python deep/clean/labels.py                  # what each cleaning rule changes, per provider
sh deep/queue.sh deep/clean/jobs.txt         # every run above, three at a time, resumable
python deep/fixed.py --tags bt bts1 lcsw lcsws1
```

## bt (2026-11-14): taught by the ball, still blind to it, 0.540

The ball may teach the model on the training matches; it may never be read when a match is
labelled. v2 used it in one weak way (a side head guessing the ball's position, shown not to
matter). `--ptr` makes the ball's lesson part of the path to the answer:

- **Who is on the ball.** After the convolutions the net points at one of the frame's players (a
  distribution over the set). On training matches the pointing is taught by the real ball: the
  target is a softmax of minus distance to the ball over 2 m. What it pointed at (that player's
  features, where he stands, and how the pointing splits between the two teams) is handed on to
  the GRU, so the label is decided with "who has it" in hand. This is the two-level design of
  Kim et al. (KDD 2023).
- **Three more ball targets** on side heads: the ball's velocity, "the ball was struck or stopped
  within 0.2 s" (its velocity jumps by more than 5 m/s between 0.2 s steps), and how far the ball
  is from the nearest player (loose or held).

Held out by provider, validation-picked decode (`decode.py`), spellF1:

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| v2, seed 0 | 0.554 | 0.633 | 0.398 | 0.529 |
| v2, seed 1 (`v2s1`) | 0.544 | 0.616 | 0.398 | 0.519 |
| control `btc`: the pointer stage, nothing of the ball in the loss (`--stage --ball 0`) | 0.552 | 0.614 | 0.397 | 0.521 |
| **bt**, seed 0 (`--ptr 0.5`) | 0.568 | 0.637 | 0.410 | **0.539** |
| **bt**, seed 1 (`bts1`) | 0.570 | 0.650 | 0.400 | **0.540** |

Both bt runs beat both v2 runs on every provider. The control says the gain is the ball's
teaching, not the extra layer. Paired by match, two seeds a side (`paired.py v2,v2s1 bt,bts1`),
bt minus v2:

| | IDSSE (7) | Metrica (3) | SkillCorner (20) | provider mean | matches up |
|---|---|---|---|---|---|
| shipped decode, labels as they are | +0.029 | +0.019 | +0.004 | 0.527 -> 0.545 | 25 / 30 |
| shipped decode, folded labels | +0.016 | +0.018 | +0.002 | 0.614 -> 0.626 | 23 / 30 |
| 2 s minimum possession, labels as they are | +0.016 | +0.014 | +0.009 | 0.524 -> 0.537 | 26 / 30 |
| 2 s minimum possession, folded labels | +0.017 | +0.018 | +0.009 | 0.628 -> 0.642 | 24 / 30 |

The standard error of the 30-match difference is 0.002 to 0.003. It clears the bar (+0.01, no
provider down) in every view. SkillCorner gains least.

- **Where the gain is** (`bylength.py`): possessions of 2 to 5 s found rise from 37 / 43 / 27% to
  43 / 46 / 29% (IDSSE / Metrica / SkillCorner); under 2 s is unchanged at 3 to 10%. Team
  accuracy moves little (0.886 -> 0.888): the ball teaching sharpens when a possession starts and
  ends more than whose it is.
- **Players missing as on BAR-ATH** (`predict.py --tag bt --out btk --knock`): 0.526, against
  v2's 0.516.
- **BAR-ATH public 5 minutes, check only, nothing chosen on it:** `deep/models/bt.pt` (all 30
  matches, 2,000 steps, seed 0) scores 0.745 with 12 possessions; v2 scores 0.826 with 14; truth
  17. That is one or two possessions apart on a window whose noise is about +-0.2, so it neither
  confirms nor contradicts the 30-match result. Over the full match the two agree on 96.0% of
  graded frames (275 against 281 possessions, median 7.1 s against 6.8 s).

- **BAR-ATH without labels** (review of #35). Per half bt gives 126 / 149 possessions and v2
  130 / 151; 13% / 13% of bt's are under 2 s (v2 14% / 11%) and 37% / 40% under 5 s (v2 42% /
  42%). Both call 44 to 48% of graded frames dead, against 20% in the public window, so the
  window's 17 possessions do not scale to the match by frame count: per live minute the window
  has 4.2 possessions and bt 4.8. The two disagree on 4.0% of graded frames, and less, not more,
  where players are missing: 2.2% with 15 or fewer tracked, 2.6% with 16 to 18, 4.2% with 21 or
  more. Half of the disagreement is bt live where v2 is dead.
- **The decode suits bt** (`compare.py --tag bt --sweep`, one setting in all folds): the shipped
  cost 4 with 1 s minimums scores 0.545, the best of the 20 settings 0.546 (cost 2). A lower
  minimum does not help: 0.543 at 0.5 s or none.
- **The pointer when the carrier is not tracked** (review of #35): the target then names the
  nearest player still tracked. With players knocked out as on BAR-ATH bt loses 0.014 (0.540 to
  0.526) and v2 loses 0.013 (0.529 to 0.516), so the pointer is no more fragile than the rest.
  Teaching it only where a tracked player is within 3 m of the ball is untried.

- **bt labels BAR-ATH from 2026-11-15** (Gambtt's choice): `run.py` defaults to `--tag bt`;
  `--tag v2` still gives the earlier labels. Two runs give byte-identical files that pass
  `evaluation/validate_submission.py`. Where bt is live and v2 dead (3,126 frames in 96
  stretches, median 0.3 s, 7 over 5 s) 56% of the frames lie within 5 s of a Smart Tagging set
  piece, against 35% of all graded frames, and the reverse disagreement is the same (57%): the
  two differ on when a restart begins and ends, not on playing through stoppages. Smart Tagging
  marks only 59 set pieces, so this is a coarse check, and it was looked at, not tuned on.

**How the no-ball rule is kept.**

- `PossessionNet.forward(x, mask)` is handed the players and nothing else; the ball enters only
  `train.ball_targets`, which is called on training batches.
- Held out: a fold's ball targets come from the two training providers only. The held-out
  provider's ball is never read by training, by prediction or by the choice of decode settings
  (those are picked on validation matches from the training providers, by their labels).
- BAR-ATH: its ball is never read. `bt.pt` is trained on the 30 outside matches.
- `tests/test_deep_no_ball.py` replaces a match's ball with blanks, noise and its own reverse and
  requires the input tensors and the predictions to be identical bit for bit, for v2 and bt.
- The labels of the outside matches were made by their providers with the ball in view; that is
  the truth being learned, as before. The folded labels in `paired.py` use the scored match's
  ball to build the yardstick, never a prediction.

Not tried here: a teacher that reads the ball, distilled into this net (the labels already carry
most of what such a teacher knows); labels cleaned with the ball.

```bash
sh deep/queue.sh deep/jobs.txt             # the runs above, three at a time, resumable
python deep/train.py --fold none --steps 2000 --ptr 0.5 --tag bt   # -> deep/models/bt.pt
python deep/run.py --tag bt --out deep_bt.csv
```

## v2 (2026-11-13): the pitch turned the right way, 0.529

`data.half_tensors` turned the pitch so home defends the left by comparing home's mean position
with halfway. A side that dominates a half averages beyond halfway, so 5 of the 60 outside halves
were turned the wrong way (4 SkillCorner, 1 IDSSE): the same bug `possession/features.own_goal`
had. On those halves v1's team accuracy was 0.70 against 0.82 on the rest of SkillCorner. Now the
two teams are compared with each other. BAR-ATH was read correctly either way.

Held out by provider, spellF1 (mean with macroF1 / alive-dead / team):

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| v1 | 0.541 | 0.635 | 0.381 | 0.519 / 0.839 / 0.935 / 0.880 |
| v1's nets, those halves predicted again (`predict.py --tag v1 --out v1o`) | 0.552 | 0.636 | 0.398 | 0.529 / 0.844 / 0.936 / 0.886 |
| **v2**: retrained with the fix, same seed | 0.554 | 0.633 | 0.398 | **0.529** / 0.845 / 0.936 / 0.886 |

The gain is the five halves being read correctly, not a better network: retraining on the
corrected halves changes nothing further. It is a fix to the same seed, so it is not subject to
the 0.013 seed-to-seed spread the variants below are. `models/v2.pt` is the all-30 model.

BAR-ATH public 5 minutes with `models/v2.pt`, nothing tuned on it (check only; 17 possessions,
so about ±0.2 of noise): spellF1 0.826, macroF1 0.918, 14 possessions against 17 (v1: 0.775,
0.894, 15). Full match: 281 possessions, 47% dead (v1: 280, 46%); the two agree on 97.6% of
graded frames.

**Regenerating the file.** `run.py` pins one thread. On the machine it was made on (Intel Xeon,
torch 2.15.0, Python 3.11) `python deep/run.py --threads 1`, `4` and `8` write byte-identical
CSVs (sha256 `5a94b7d8...0d813d`). It has not been checked on another CPU, where the last bits of
a probability could differ and a near-tie in the decode could move a boundary. `models/v2.pt`
came from one uninterrupted run of `train.py --fold none --steps 2000 --tag v2` (4 threads, seed
0); checkpoints now carry torch's generator as well, so a resumed run would match too. `v1.pt` is
no longer shipped: v2 replaces it, and `predict.py --tag v1` shows what the fix alone does.

**Against `hmm` v7 on BAR-ATH, no labels read** (`compare.py --against evaluation/results/hmm_v7.csv`,
asked for in the review of #29):

| | `deep` v2 | `hmm` v7 |
|---|---|---|
| half 1 | 130 possessions, median 7.4 s, 45% dead | 132, 8.1 s, 42% |
| half 2 | 151 possessions, median 6.3 s, 48% dead | 157, 6.2 s, 47% |
| public 5 minutes (check only, 17 possessions) | spellF1 0.826, macroF1 0.918, 14 possessions | 0.606, 0.783, 16 |

The two agree on 86.5% of graded frames: 84.5% where the ball is seen (59% of frames) and 89.4%
where it is missing. Where both call the ball live they name the same team on 89.4%. The
disagreements are 5.3% of frames on the team (93% of those with the ball in view), 5.1% where
`deep` says dead and `hmm` live, and 3.0% the other way. So the two count about the same number
of possessions (281 and 289) and differ mostly on who has a ball that the tracker can see, which
is where a combination has something to decide. The public window favours `deep` on macroF1 by
more than its spellF1 noise allows us to say for spells.

**Is the decode merging possessions?** (`compare.py --sweep`) The review noted 281 possessions
against about 350 if the whole match ran at the public window's density. Every setting, the same
in all three folds, with BAR-ATH's count beside its held-out score:

| switch cost | shortest run kept, H/A and D | IDSSE | Metrica | SkillCorner | mean | BAR-ATH possessions (median) |
|---|---|---|---|---|---|---|
| 0 | none | 0.537 | 0.608 | 0.355 | 0.500 | 403 (3.8 s) |
| 1 | none | 0.552 | 0.630 | 0.380 | 0.521 | 332 (5.3 s) |
| 1 | 0.5 s | 0.554 | 0.633 | 0.383 | 0.523 | 327 (5.4 s) |
| 2 | 0.5 s | 0.553 | 0.633 | 0.388 | 0.525 | 316 (6.0 s) |
| 2 | 1 s | 0.554 | 0.634 | 0.394 | 0.528 | 304 (6.3 s) |
| **4 (shipped)** | **1 s** | 0.549 | 0.635 | 0.398 | 0.527 | 286 (6.8 s) |
| 8 | 1 s | 0.544 | 0.630 | 0.405 | 0.526 | 269 (7.6 s) |
| 4 | 2 s and 3 s | 0.532 | 0.621 | 0.420 | 0.524 | 235 (8.6 s) |

(286 here is 281 once the frames before each kickoff are set to D.) From 235 to 332 possessions
the held-out mean stays within 0.521 to 0.528, inside the 0.013 seed spread, so the outside
matches cannot say how finely to cut: IDSSE wants a cheaper switch, SkillCorner a dearer one.
Getting to 350 needs no decode at all and costs 0.03. `hmm` v7 lands on 289 from the ball, by a
different route, and 350 is five minutes extrapolated to a match. So the shipped setting stays;
cost 2 with 1 s runs (304 possessions, 0.528) is as good a choice, and nothing here should be
picked on the public window.

**Players missing as on BAR-ATH** (`predict.py --knock`: `external/score.py drop_players`, about
7% of player-frames lost in runs and bursts, speeds recomputed):

| | full tracking | players knocked out |
|---|---|---|
| v1, direction fixed | 0.529 | 0.511 |
| v2 | 0.529 | 0.516 (IDSSE 0.547, Metrica 0.610, SkillCorner 0.392) |
| v3: v2 with half the training windows knocked out the same way (`train.py --knock 0.5`) | 0.524 | 0.516 (0.528, 0.624, 0.396) |

BAR-ATH's gaps cost about 0.013 to 0.018, the same as they cost `possession/` (0.016). Training
on knocked-out windows does not win it back (0.516 either way, and 0.005 lower on full
tracking), so v3 is not kept. The loss is information that is gone, not a network surprised by
gaps: v1 already hides up to 15% of players at random in training.

**With `possession/` in the same decode** (`stack/`): 0.530 against 0.529. The hand-built signals
add nothing the network has not learned, so this model stands alone as the players-only model.

## A fourth tracking provider and tracker-style changes (2026-11-14): no gain from either

The design review put the cost of an unseen tracking provider at about 0.06, and BAR-ATH is one.
Two things were tried: more providers, and making our matches look like other providers'.
Neither raises the held-out mean. Nothing here changes `models/v2.pt`.

**SoccerTrack v2 as a fourth provider** (`external/build_soccertrack.py`; Bepro tracking of 8
university matches with the provider's per-frame ball status; CC BY 4.0, see `external/README.md`).
The files sit in `data/soccertrack/` and are used only with `train.py --with soccertrack`;
without the flag the training set is v2's whatever is on disk. With it SoccerTrack joins the
training providers of every fold; the held-out provider is as before. 1,500 steps, one fixed
decode for every tag (`deep/fixed.py`: cost 4, 1 s / 1 s on the labels as built; cost 4,
2 s / 3 s on the labels folded to the control rule). Equal-provider means:

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| labels as built: v2, seed 0 / seed 1 | 0.549 / 0.555 | 0.635 / 0.628 | 0.398 / 0.398 | 0.527 / 0.527 |
| + SoccerTrack, an equal third of windows (`st`, seed 0) | 0.531 | 0.627 | 0.422 | 0.526 |
| + SoccerTrack and `--style 0.5` (`sty`, seed 0) | 0.518 | 0.617 | 0.419 | 0.518 |
| + SoccerTrack at 15% of windows (`stw`, `--share soccertrack=0.15`), seed 0 / seed 1 | 0.548 / 0.544 | 0.649 / 0.614 | 0.414 / 0.410 | 0.537 / 0.523 |
| folded labels: v2, seed 0 / seed 1 | 0.672 / 0.679 | 0.723 / 0.710 | 0.489 / 0.487 | 0.628 / 0.626 |
| `st`, seed 0 | 0.657 | 0.711 | 0.515 | 0.628 |
| `sty`, seed 0 | 0.650 | 0.709 | 0.509 | 0.623 |
| `stw`, seed 0 / seed 1 | 0.670 / 0.664 | 0.727 / 0.701 | 0.505 / 0.503 | 0.634 / 0.623 |

How to read it. The 15% share was the second share tried and was picked on these held-out
matches, and its first seed looked like a gain (+0.010 / +0.006). The second seed, run because
the review of this PR asked for it, does not repeat it (-0.004 / -0.003): over the two seeds
`stw` is 0.530 against 0.527 on the labels as built and 0.629 against 0.627 folded, which is
nothing. What does repeat is the shape: SkillCorner goes up in every SoccerTrack run (+0.012 to
+0.026) and IDSSE never does (-0.001 to -0.018). SoccerTrack's long possessions pull the model
towards SkillCorner's way of cutting possessions and away from DFL's. On unseen matches of the
training providers (`fixed.py --kind val`) `stw` equals v2 (0.582 / 0.669 against 0.583 / 0.670)
and `st` is 0.01 below. So SoccerTrack is not folded into the base model. Fold nets and
probabilities: `/mnt/project-files/deep/{st,sty,stw,stws1}/`.

**The model on a tracker it has never seen.** v2's fold nets, which never saw Bepro tracking,
score 0.68 to 0.75 spellF1 on two SoccerTrack matches (alive/dead 0.96 to 0.97, team 0.94),
against 0.40 to 0.64 on their held-out providers. With SoccerTrack in training the same matches
score 0.76 to 0.77 (the validation match; one match, one seed). So a new tracker in itself costs about 0.04 here, and
how a provider cuts possessions matters far more than how it tracks: Bepro's possessions have a
median of 9 to 10 s, DFL's 3.7 s.

**Which tracking differences the model feels** (`deep/shift.py --tag v2`: v2's fold nets on
their held-out matches with the tracking changed one way at a time, `deep/trackers.py`; spellF1
on labels as built / folded, provider mean over 11 matches):

| change | as built | folded |
|---|---|---|
| as recorded | 0.519 | 0.621 |
| smoothed 0.4 s / 1.2 s | 0.518 / 0.511 | 0.614 / 0.612 |
| 10 fps or 5 fps underneath | 0.519 | 0.621 |
| drift of 0.5 m over 3 s | 0.516 | 0.616 |
| pitch 4% too long / shifted 1 m both ways | 0.522 / 0.522 | 0.623 / 0.619 |
| broadcast view 60 m wide | 0.518 | 0.617 |
| 7% / 15% of player-frames lost in 5 s runs | 0.517 / 0.513 | 0.617 / 0.610 |
| independent noise of 0.1 m / 0.3 m per frame | 0.331 / 0.241 | 0.390 / 0.298 |
| positions on a 1.05 x 0.68 m grid | 0.289 | 0.338 |

The model does not care about smoothing, frame rate, calibration, a narrower view or lost
players. It cares a great deal about frame-to-frame jitter, which goes straight into the
velocities; but no provider has any to speak of. Median second difference of position at 25 fps:
IDSSE 1.0 cm, Metrica 0.2 to 0.3 cm, SkillCorner 0.2 cm, BAR-ATH 0.15 cm. Only SoccerTrack's raw
grid would trip it, which is why the converter smooths it. So training on changed tracking
(`train.py --style`) had nothing to fix, and `sty` above confirms it: slightly worse than `st`.
`--style` stays in the code as an option and is not used.

The median hides tails, so BAR-ATH's were checked too (no labels read): a player moves more than
0.5 m between two frames in 2 to 7 of 10,000 player-frames (Metrica's sample matches 2, the
others 0), and sits more than 0.1 m off the line between his neighbouring frames in 0.1 to 0.2
of 10,000 (Metrica 2 to 3, IDSSE 0.4). Tracks start anew about 11 times a minute, as on
SkillCorner (14). So BAR-ATH has no jitter for the model to trip over; its difference is tracks
that break, which the missing-player work in `adapt/` covers.

## v1 (2026-11-12)

Held out by provider, the same folds and equal-provider mean as `external/heldout.py`:

| held out | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE (7) | 0.541 | 0.877 | 0.949 | 0.895 |
| Metrica (3) | 0.635 | 0.845 | 0.912 | 0.929 |
| SkillCorner (20) | 0.381 | 0.796 | 0.945 | 0.814 |
| **mean** | **0.519** | 0.839 | 0.935 | 0.880 |

For comparison, same protocol: `possession/` on main 0.282; the fitted team model with Viterbi on
`claude/project-thread-3lcb21` 0.435; the classical research branch 0.435 to 0.437.

Each fold trains on the other two providers minus a few validation matches (1 IDSSE, 1 Metrica,
3 SkillCorner where they are in training), and the decode settings are chosen on those validation
matches. The SkillCorner fold therefore learns from only 8 matches, which is the likely reason it
is the weakest; the model that labels BAR-ATH learns from all 30.

BAR-ATH, nothing tuned on it (check only; 17 possessions, so about ±0.2 of noise on spellF1):

| public 5 minutes | spellF1 | macroF1 | alive/dead | team acc | possessions |
|---|---|---|---|---|---|
| `deep` | 0.775 | 0.894 | 0.961 | 0.968 | 15 (truth 17) |
| `possession` (main) | 0.352 | 0.826 | 0.960 | 0.872 | 8 |

Full match: 130 / 150 possessions per half, median 7.8 / 6.6 s, 44% / 48% dead.

## What it is, and where it comes from

- **Players as an unordered set.** Each frame's players go through two self-attention blocks and
  are pooled per team (Set Transformer, Lee et al. 2019; used for exactly this in Kim et al., KDD
  2023, "Ball trajectory inference from multi-agent sports contexts using set transformer and
  hierarchical Bi-LSTM"). No player identity is needed, which suits our 46 swapping track slots.
  Per player: position, velocity, speed, change of speed, team.
- **Time.** Twelve dilated convolutions (0.2 s to 6.4 s steps, twice) and a two-way GRU over the
  5 fps sequence, as the offline task allows (MS-TCN, Farha & Gall 2019; TranSPORTmer, Capellera
  et al. 2024, which labels possession states jointly with trajectories).
- **The ball as a training target.** A second head predicts where the ball is, on live frames
  where it was seen. Kim et al. and TranSPORTmer both find the ball is largely recoverable from
  the players; here that knowledge is forced into the shared features and the head is thrown away.
  A third head marks frames within 0.4 s of any change of label.
- **Symmetry.** The pitch is turned so home defends the left; swapping the teams and mirroring
  the pitch are exact symmetries, used as augmentation and averaged over at prediction.
- **Robustness to tracking.** Position jitter (0.2 m) and hiding up to 15% of players per window.
- **Decode.** A three-state chain with one switching cost, then runs under 1 s absorbed. The
  score is flat across the decode grid (IDSSE 0.527 to 0.541, Metrica 0.630 to 0.636), so the
  network, not the decode, carries the result.

1,500 training steps per fold (2,000 for the final model) of 16 one-minute windows, CPU only,
about 70 minutes per fold on one core.

## What was tried on top (same folds, 1,500 steps, one seed each)

| variant | IDSSE | Metrica | SkillCorner | mean | kept? |
|---|---|---|---|---|---|
| **v1** (above) | 0.541 | 0.635 | 0.381 | **0.519** | yes |
| v1 again, another seed (`--seed 1`) | 0.520 | 0.618 | 0.380 | 0.506 | the same model: this is the run-to-run noise |
| no ball target (`--ball 0`) | 0.532 | 0.608 | 0.380 | 0.507 | no: inside the seed-to-seed spread, so the ball target's worth is not shown |
| v1 and no-ball probabilities averaged | 0.539 | 0.623 | 0.391 | 0.518 | no: nothing over v1 |
| short possessions weighted up in the loss (`--spell 0.5`) | 0.552 | 0.653 | 0.371 | 0.525 | no: +0.006, SkillCorner down 0.010 |

Two seeds of the same model differ by 0.013 in the mean (0.519 and 0.506), and up to 0.021 on a
fold. So none of the variants is distinguishable from v1, including the one without the ball
target, and the honest headline for this model is about 0.51, give or take 0.01. The gap to the
hand-built models (0.435) and the baseline (0.282) is many times that.

**Where v1 still loses.** Held out, share of true possessions matched at IoU 0.5, by length:

| | under 2 s | 2 to 5 s | 5 to 10 s | over 10 s | possessions predicted / true |
|---|---|---|---|---|---|
| IDSSE | 8% | 36% | 63% | 87% | 1,813 / 3,061 |
| Metrica | 10% | 44% | 67% | 81% | 714 / 1,041 |
| SkillCorner | 3% | 25% | 47% | 66% | 6,409 / 6,500 |

Long possessions are found; ones under 2 s almost never are, the same wall the hand-built model
hit. Weighting them up in the loss did not move it, so from players alone they may not be there
to find. BAR-ATH's public window has 2 of 17 possessions under 2 s, against 20 to 30% outside.

## Caveats

- The model that labels BAR-ATH is a single run (seed 0); its held-out twin scored 0.519, the other seed 0.506.
- The outside labels are not the organisers' rule (`external/README.md`); 0.519 is not a
  leaderboard prediction.
- `torch` is a new dependency, locked in `uv.lock` under the `deep` group (`uv sync --group deep`).
- A learned model can pick up a provider's tracking quirks. Holding a provider out guards
  against that, and the BAR-ATH public window agrees, but that window is five minutes.
