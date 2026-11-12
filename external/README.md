# Out-of-sample matches

The public 5 minutes may not be used to tune (RULES.md §2), so we validate on **other games**, which
RULES.md allows in any amount. Thirty labelled matches, in the same arrays `possession/load.py` gives
for ours:

```bash
uv run python external/build.py            # -> data/external/*.npz (gitignored, ~1 min/match)
uv run python external/score.py --seen 0.59 [param=value ...]
uv run python external/score_hmm.py --seen 0.59   # the HMM decoder in hmm/, and smooth50 (all three sets)
```

| set | matches | tracking | label source |
|---|---|---|---|
| `idsse_*` | 7 (2 Bundesliga, 5 2. Bundesliga, 2022-23) | TRACAB optical, 25 fps, ball ~always seen | DFL's per-frame `BallStatus` / `BallPossession` |
| `metrica_*` | 3 anonymised sample games | Metrica, 25 fps, ball seen ~60% (like ours) | rebuilt from events, see `build.py` |
| `skillcorner_*` | 20 (Australian A-League, 2024-25) | SkillCorner broadcast, 10 fps resampled to 25, ball seen ~94% of labelled live frames | phases of play + per-frame possession group |

Where the raw files come from (put them under `data/external/raw/`):

- IDSSE: [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0,
  Bassek et al., *Scientific Data* 12, 195 (2025). The figshare originals sit behind a bot check.
- Metrica: `git clone https://github.com/metrica-sports/sample-data` into `raw/metrica/`.
- SkillCorner: [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT; cite as
  SkillCorner Open Data (A-League 2024-25). List it with the other two in the technical document. The tracking
  files are Git LFS (~90 MB each, 2 GB in all). Put `data/matches.json` and each
  `data/matches/<id>/` folder under `raw/skillcorner/`, fetching `*_tracking_extrapolated.jsonl`
  from `media.githubusercontent.com` rather than `raw.githubusercontent.com`.

## What the labels are, and are not

Neither label source uses the challenge's own "controlled possession" rule, so absolute scores
here are not leaderboard predictions. Use them to compare two versions of the model.

- **DFL `BallPossession`** is set by data operators and likely flips on the first touch more than
  our rule does.
- **Metrica events:** dead from a ball-out, foul or shot-out until the next on-ball action. While
  alive, possession belongs to the team of the last pass, carry, shot, recovery or set piece.
  Challenges and lost balls don't change it.
- **SkillCorner:** dead outside its phases of play, which exist only while the ball is in play.
  Inside a phase, the team SkillCorner names in possession, held through frames where it names
  nobody (passes, loose balls). Extrapolated ball positions are dropped. Extrapolated players are
  kept, because our feed tracks almost everyone. Broadcast cutaways, such as replays, track nobody;
  those frames are labelled X, so they are not graded. That is about a quarter of each match and
  half the dead time, and it includes the hardest dead-ball stretches. A SkillCorner score is
  therefore not comparable with a full-match score.
- **SkillCorner timing:** labels are taken from the previous 10 fps frame, so a change of
  possession can be up to 0.1 s late. Fine for training; don't tune boundary timing on this set.

**DFL's spells are short.** A match has about 510, with a median of 3 s. Our 5 public minutes
have 17 possessions, roughly 360 over a full match, so DFL counts some touches we would not.
Metrica has about 315, and SkillCorner about 300, which is closer. Expect a low
spellF1 on IDSSE even when the model is right by the organisers' rule. That set is better for
alive/dead and team accuracy.

| possessions (H/A runs) | per graded minute | mean | median | 90th pct | D share |
|---|---|---|---|---|---|
| IDSSE | 4.6 | 7.2 s | 3.7 s | 18.6 s | 45% |
| Metrica | 3.6 | 11.3 s | 5.9 s | 28.9 s | 32% |
| SkillCorner | 4.4 | 9.6 s | 5.8 s | 23.1 s | 29% |

All three cut possessions shorter than the public window (17 in 5 minutes, mean 14.2 s, median
7.6 s, D 20%), so a model fitted to them may lean towards short possessions. These are descriptive
only: the labels are not adjusted towards the public window.

## Baseline: `possession/` with the stoppage model (players only: more spread-out team, keepers left out, has it; a logistic model on four player-speed signals for dead; short runs absorbed)

Held out by provider (`external/heldout.py`: the stoppage weights refitted on the other two
providers each time), the number to compare changes on:

| held out | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE (7) | 0.217 | 0.722 | 0.883 | 0.767 |
| Metrica (3) | 0.299 | 0.715 | 0.865 | 0.803 |
| SkillCorner (20) | 0.284 | 0.700 | 0.889 | 0.799 |
| mean of the three | 0.266 | 0.712 | 0.879 | 0.790 |

`score.py` in-sample (weights fitted on all 30): IDSSE 0.227, Metrica 0.300, SkillCorner 0.285,
all 30 0.273 spellF1. Before the stoppage model (median speed under 1.2 m/s): held-out mean 0.250,
all 30 0.260. The experiment log is in `possession/README.md`.

`score.py` prints each set's mean as well as the overall one; compare like with like. SkillCorner
is not graded on its cutaways (see above), so its row is not comparable with the others. The IDSSE
and Metrica rows are unchanged by measuring `--seen` on labelled frames.

Before the short runs were absorbed (`min_team=0 min_dead=0`): 0.187 / 0.212 / 0.194 spellF1.

It never reads the ball, so `--seen` doesn't change these. The ball-based nearest-team v0 before it
scored 0.057 spellF1 on all 10 at `--seen 0.59`.

The old v3 model these numbers used to describe (0.431 across all 10, ball-based) was never committed
and is lost. It is still the best single-model score seen here: this v0 is a players-only lens
meant to be combined with `hmm/`, not a replacement for it.

## The HMM decoder (`hmm/`), `score_hmm.py --seen 0.59`

IDSSE and Metrica are each scored with the speed, flight and gap-distance tables counted on the
other one. SkillCorner is held out: it is scored with the decoder's shipped constants, which were
counted on IDSSE and Metrica only, and it is never used to count those tables. `hmm_v6`'s width
table is counted, for every set, on the labels of the other sets, each counting once.

| spellF1 | smooth50 | hmm_v2 | hmm_v3 | hmm_v4 | hmm_v5 | hmm_v6 |
|---|---|---|---|---|---|---|
| IDSSE mean (7) | 0.227 | 0.183 | 0.223 | 0.354 | 0.418 | 0.441 |
| Metrica mean (3) | 0.426 | 0.390 | 0.443 | 0.650 | 0.742 | 0.753 |
| all 10 | 0.287 | 0.245 | 0.289 | 0.443 | 0.516 | 0.535 |
| SkillCorner mean (20), `--seen 0.59` | 0.226 | 0.285 | 0.298 | 0.337 | 0.333 | 0.354 |
| SkillCorner mean (20), ball as recorded (`--seen 1.0 --sets skillcorner`) | 0.264 | 0.320 | 0.328 | 0.374 | 0.359 | 0.369 |

SkillCorner is not graded on its cutaways (see above), so its rows are not comparable with the
others and are not pooled with them. Its ball is seen on about 80% of labelled frames as recorded;
`--seen 0.59` knocks it down to ours, measured on labelled frames.

**v6 is v5 plus the width clue** (`hmm.clues.team_width_difference`: the team spread wider across
the pitch has the ball), at a quarter of the other clues' weight (`WIDTH_WEIGHT = 0.05`). It reads
the players only. `hmm_v6 - hmm_v5` in spellF1, paired by match:

| | mean | standard error | worst match | matches up |
|---|---|---|---|---|
| IDSSE | +0.023 | 0.003 | +0.014 | 7 of 7 |
| Metrica | +0.011 | 0.005 | +0.002 | 3 of 3 |
| SkillCorner, `--seen 0.59` | +0.021 | 0.003 | +0.003 | 20 of 20 |
| SkillCorner, ball as recorded | +0.009 | 0.002 | -0.012 | 18 of 20 |

The gain is in team accuracy (IDSSE 0.855 to 0.874, SkillCorner 0.805 to 0.831 at `--seen 0.59`);
alive/dead does not move. It is larger where the ball is missing more, which is what a
players-only clue should do. At a weight of 0.2 the clue overrides the ball and costs Metrica
0.061. `--tables` prints the width table counted on all three sets, each counting once; it matches
`WIDTH_GIVEN_HOME` and `WIDTH_GIVEN_DEAD` in `hmm/decode.py` to 0.001, and no single set's home
row is further than 0.024 from it (dead row 0.031).

**The width weight is tuned on these matches.** spellF1 gain over v5 by weight (SkillCorner with
its ball as recorded):

| weight | 0.02 | 0.035 | 0.05 | 0.075 | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|---|---|---|
| IDSSE | +0.017 | +0.020 | +0.023 | +0.024 | +0.025 | +0.015 | +0.006 |
| Metrica | +0.007 | +0.007 | +0.010 | +0.006 | 0.000 | -0.028 | -0.063 |
| SkillCorner | +0.006 | +0.008 | +0.010 | +0.012 | +0.016 | +0.018 | +0.018 |

Chosen with each set left out in turn, the other two pick 0.05, 0.1 and 0.05, and the set left out
then gains +0.023 (IDSSE), 0.000 (Metrica) and +0.010 (SkillCorner). That is +0.011 on average,
and it is the fair size of this clue: below the 0.03 bar v5 was held to. v6 is kept because no
set loses at 0.05 and the direction is the same on all three, not because the gain is large.

**Does it just favour the wider team? Checked on BAR-ATH, without labels.** Home is the wider team
on 56% of graded frames there (median difference +1.4 m and +1.1 m by half). That mixes style with
possession: v5 gives home 64% and 57% of live frames, and the midpoint between the median width
on v5's home frames and on its away frames is +1.1 m and +0.7 m, about half a bin. The 10 IDSSE
and Metrica matches have per-match medians from -0.6 to +1.7 m. From v5 to v6, 6.5% of graded
frames change: 4,012 from H to A and 3,433 from A to H, a net +0.2% towards home, and home's share
of live frames is 64.1% and 56.2% (v5: 64.2% and 56.7%). So the changes are not piled on one team.
Ball-missing frames change more often (8.3%) than ball-visible ones (5.2%), as intended.
Subtracting each half's median width first changes a further 2.7% of frames, every one of them
away from home, because the median also removes home's real share of the ball; on the external
matches centring moved spellF1 by 0.003 at most. The clue is left uncentred.

A team's width needs 6 of its players tracked (`MIN_WIDTH_PLAYERS`). On BAR-ATH 6% of graded
frames have a team below that and the clue is silent there; the external scores are the same to
three decimals with a floor of 3.

The constants `WIDTH_GIVEN_HOME`, `WIDTH_GIVEN_DEAD` and `WIDTH_WEIGHT` are hard-coded, so the
submission is reproducible from the repo alone, but their derivation needs the 30 outside matches.
The technical document must list all three sources (above) as assets used.

v5 beats v4 on each of the 10 (by 0.032 to 0.112). Its two changes one at a time, then the variant
that is not the default, with the possessions predicted against the true count:

| spellF1 (possessions) | v4 | + weight 0.2 (`hmm_v4w`) | + gap-distance (v5) | v5 + fitted place (`hmm_v5fit`) | truth |
|---|---|---|---|---|---|
| IDSSE mean (7) | 0.354 (197) | 0.394 (262) | 0.418 (253) | 0.414 (248) | (437) |
| Metrica mean (3) | 0.650 (219) | 0.701 (273) | 0.742 (274) | 0.774 (286) | (347) |

The weight and the gap-distance clue help on both providers, and they are v5. The fitted place
table (`python -m hmm.decode --fit-place`) helps only on the Metrica samples (+0.032, 3 of 3
matches up) and costs a little on IDSSE (-0.004, 1 of 7 up), whose ball is knocked out at random
and has nothing to fit. Over all 10 that is +0.007, with a 5-95% interval over matches of -0.002
to +0.016. With nothing to separate them, the default is the hand-set table.

**SkillCorner prefers the lower clue weight.** v5 scores below v4 there (0.359 against 0.374 with
the ball as recorded), the opposite of IDSSE and Metrica, where weight 0.2 beat 0.1 by 0.043. With
v5's clues on SkillCorner: 0.377 at weight 0.1, 0.368 at 0.15, 0.359 at 0.2 and 0.345 at 0.3.
`WEIGHT` stays 0.2. Averaged over all 30 matches v5 scores 0.411 at 0.2 against 0.399 at 0.1, and
SkillCorner is broadcast tracking at 10 fps that is not graded on its cutaways, so it is the set
least like ours.

**Where both still go wrong: the team either side of a stoppage.** The truth has the same team
before and after 27% of its stoppages (IDSSE 29%, Metrica 21%); `hmm_v5` and `hmm_v5fit` have it at
49% and 47%. Few of those are stoppages that are not there (5 to 8 a match). Most are real
stoppages where the model has the wrong team on one side: 51 same-team of the 97 it shares with
the truth on IDSSE, where the truth has 28.

**How much each kind of mistake is worth.** v5 over the original 10 (0.516), given part of the
truth: the true restarting team after each stoppage, +0.010; the true stop and restart instants,
+0.062; the true alive/dead on every frame, +0.091; the true team on every live frame, +0.287. So
most of what is left is naming the team in open play, not the stoppages. Three ideas aimed at
stoppages were tried and dropped, each moving spellF1 by 0.005 or less: states for the restart
phase, a stoppage that remembers the last team, and simulated team anchors.

**Neither IDSSE nor Metrica can validate a clue that reads the ball during a stoppage.** IDSSE
freezes the ball while play is dead: it moves exactly zero on 100% of dead frames. Metrica shows
no ball on dead frames (seen on 0-8% of them). A ball-speed clue gained +0.051 on IDSSE for that
reason alone and was dropped. Our ball does neither: on BAR-ATH it is usable on 26% of frames in
the 5 s before a tagged set piece, and still (under 0.5 m/s) on 3% of those.

Read the two sets differently:

- **IDSSE's knocked-out ball is not how ours goes missing.** The knockout makes 12 short gaps a
  minute (median 1.4 s), in play and out alike. BAR-ATH has 1.4 a minute with a median of 10.7 s.
- **The Metrica samples' ball sits on the players** (a median 0.3 m from the nearest one, against
  1.9 m on BAR-ATH), and it is missing on 92-100% of dead frames and 9-13% of live ones. Their
  labels are rebuilt from the same events that ball follows, which flatters any touch-based clue.
- With smooth50 and the other three starter baselines, IDSSE reproduces the organisers' published
  full-match spellF1 (0.04 / 0.16 / 0.23 / 0.20 against 0.05 / 0.20 / 0.24 / 0.22) and Metrica does
  not (0.10 / 0.37 / 0.43 / 0.32). So take the level from IDSSE and the direction from both.

Use `--seen` on IDSSE and SkillCorner, which is measured on labelled frames. With the ball always
visible, the model's hardest problem disappears.
But `--seen` drops the ball at random, live or dead alike, and our match does not lose it that way.
With the ball missing, BAR-ATH's players move at a median 1.0 m/s against 1.9 m/s with it seen,
which is the dead-ball pattern. Metrica shows the same pattern: there the ball is missing on 92% of
dead frames and 12% of live ones. So IDSSE under `--seen` punishes "missing means dead" more than
our match would.
Scores also get no help from set-piece tags, because these matches have no Smart Tagging.

## The learned labeller (`seq/`), `score_seq.py --seen 0.59`

Boosted trees score each frame H / A / D from team-level features, and a chain decodes the half
(`seq/model.py`, after Kim et al., KDD 2023, and PathCRF, arXiv 2602.12080). Each set is scored by a
scorer fitted on the other two sets only, the two counting equally. Needs scikit-learn
(`pip install -r requirements-seq.txt`, pinned); one run is six fits and takes about 40 minutes,
cached under `data/cache/seq/`. A fit at 1, 4 and 8 threads gave byte-identical probabilities;
the thread count is fixed anyway (`seq.model.THREADS`).

"hybrid" is `hmm_v6` with the labeller's per-frame log-probabilities added to its clues at a
weight. At weight 0 it reproduces the `hmm_v6` row above exactly. The labeller scores every 5th
frame and each score is held for 5 frames, so the weight is per 25 fps frame.

**The chain and the weight are chosen for a set without any of its labels**: on each of the other
two sets, as scored by a scorer fitted on the third set alone. That picks a weight of 0.05, 0.1
and 0.05. One path is left open for the weight: the `hmm_v6` tables under the hybrid are the ones
`score_hmm.py` uses (IDSSE's counted on Metrica and the reverse, width on the other sets), so the
emissions a set's weight is chosen on were counted partly from that set's labels. It is one
weight among five, and the same tables sit under the `hmm_v6` column:

| spellF1 | hmm_v6 | seq alone | hybrid | hybrid - hmm_v6, paired (standard error) | worst match | matches up |
|---|---|---|---|---|---|---|
| IDSSE mean (7) | 0.441 | 0.474 | 0.494 | +0.053 (0.003) | +0.039 | 7 of 7 |
| Metrica mean (3) | 0.753 | 0.637 | 0.776 | +0.022 (0.013) | -0.001 | 2 of 3 |
| SkillCorner mean (20) | 0.354 | 0.414 | 0.407 | +0.053 (0.003) | +0.027 | 20 of 20 |

Metrica's gain is not established: three matches, one of them level, and inside the 0.04 band.

The nested choice leans to low weights, because a scorer fitted on one set is weaker than the
two-set scorer whose weight it is choosing. Every weight, with the reported (two-set) scorers:

| hybrid spellF1 by weight | 0 (`hmm_v6`) | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| IDSSE | 0.441 | 0.494 | 0.517 | 0.536 | 0.537 |
| Metrica | 0.753 | 0.769 | 0.776 | 0.779 | 0.760 |
| SkillCorner | 0.354 | 0.407 | 0.411 | 0.410 | 0.398 |

No set loses at any weight up to 0.4, and 0.1 to 0.2 is the flat top on all three. Reading the
weight off this table is in-sample by one number in five.

Most of the hybrid's gain is in alive/dead (at 0.1: IDSSE 0.903 to 0.948, SkillCorner 0.903 to
0.936); team accuracy moves by about 0.01 or less. So the labeller is a better stoppage detector
more than a better judge of which team has the ball. The outside sets lose the ball at random
(`drop_ball`), not around restarts as BAR-ATH does, which is the main risk to this gain carrying
over and is not yet tested.

**The ball-alone features (`k_`) are left out by default.** With them in (`--drop "" --tag all`),
each column with its own nested choice:

| spellF1 | seq alone, with k_ | seq alone, without | hybrid, with k_ | hybrid, without |
|---|---|---|---|---|
| IDSSE | 0.474 | 0.474 | 0.510 | 0.494 |
| Metrica | 0.723 | 0.637 | 0.800 | 0.776 |
| SkillCorner | 0.359 | 0.414 | 0.383 | 0.407 |

A scorer fitted on IDSSE and Metrica learns that a still or missing ball means a stoppage, which
is how those two providers record one (above), and held-out SkillCorner pays for it: its frame
accuracy is 0.71 with them and 0.80 without. Metrica gains from them because its labels and its
ball come from the same events. Players only (`--drop b_,k_ --tag players`) scores 0.405 / 0.520 /
0.389 alone and 0.488 / 0.738 / 0.398 as the hybrid, which is below `hmm_v6` on Metrica.

Not settled: the feature groups were compared on all three sets, and `b_touch_since` still reads
a ball that IDSSE freezes. Nothing in `seq/` has tests yet. **No script in the repo fits on all 30
matches and labels BAR-ATH, so none of this is in the submission.** `seq/features.py` now takes
the (frames, slots) team map `hmm/clues.team_by_frame` gives, so that path can use the same keeper
correction as `hmm_v6`. Several features (`p_tracked`, the defensive lines, the 6-player floor)
depend on tracking quality the outside sets do not show, and how they behave on BAR-ATH's
fragmented slots is unchecked. `uv.lock` does not include the `seq` group yet.
