# Out-of-sample matches

The public 5 minutes may not be used to tune (RULES.md §2), so we validate on **other games**, which
RULES.md allows in any amount. Thirty labelled matches, in the same arrays `possession/load.py` gives
for ours:

```bash
uv run python external/build.py            # -> data/external/*.npz (gitignored, ~1 min/match)
uv run python external/score.py --seen 0.59 [param=value ...]
uv run python external/score_hmm.py --seen 0.59   # the HMM decoder in hmm/, and smooth50 (all three sets)
uv run python external/score_control.py           # hmm_v8 against hmm_v6, raw and folded labels (below)
uv run python external/score_fuse.py --tag bt,b2  # hmm_v9: v8 plus the players-only net (below)
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

- SoccerTrack v2 (added 2026-11-14, `soccertrack_*`, 8 matches): Scott, Uchida, Kuroda, Kim and
  Fujii, "SoccerTrack v2", arXiv 2508.01802, <https://atomscott.github.io/SoccerTrack-v2/>,
  dataset CC BY 4.0 (any use with credit; cite the paper in the technical document). University
  matches tracked by Bepro from a panoramic camera, 25 fps, 22 players in almost every frame,
  positions on a 1.05 m x 0.68 m grid (smoothed by the converter). Built by
  `python external/build_soccertrack.py --raw <folder>` into `data/soccertrack/` (its own folder, so
  the 30 above stay the held-out set of every other script) from `raw/<match>/<match>_tracker_box_data.xml`
  and `_metadata.xml`, about 220 MB a match, from the gated Hugging Face dataset or the authors'
  Google Drive mirror. Labels are Bepro's per-frame ball status (HOME / AWAY / BALLOUT, and
  NEUTRAL held by the team before it): 235 to 330 possessions a match, median 7 to 12 s, D 26 to
  38%. Matches 132831 and 132877 carry no ball status and are left out. `build.py`,
  `heldout.py` and the `score*.py` scripts do not use this set; `deep/train.py` does.
- Looked at and not used: PFF / Gradient Sports World Cup 2022 (64 matches of broadcast tracking,
  free on a sign-up form, no published licence); Last Row (19 goal clips); Alfheim (one team, no
  ball); SoccerNet tracking (needs an NDA).

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

## Baseline: `possession/` (players only: a fitted team model on shape and movement, a fitted stoppage model, Viterbi over H/A/D)

Held out by provider (`external/heldout.py`: both models' weights refitted on the other two
providers each time), the number to compare changes on:

| held out | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE (7) | 0.424 | 0.837 | 0.932 | 0.864 |
| Metrica (3) | 0.506 | 0.800 | 0.889 | 0.899 |
| SkillCorner (20) | 0.375 | 0.770 | 0.922 | 0.829 |
| mean of the three | 0.435 | 0.802 | 0.914 | 0.864 |

The paragraphs below describe the model before 2026-11-12 (spread lean, three stoppage signals,
short runs absorbed: held-out mean 0.282 / 0.736 / 0.898 / 0.790). The experiment log is in
`possession/README.md`.

Before the acceleration signal: held-out mean 0.266 / 0.712 / 0.879 / 0.790. With all five signals (before dropping the two redundant ones): 0.282 / 0.735 / 0.898 / 0.790.

`errors.py` breaks the held-out mistakes down by kind (wrong team near a turnover or mid-possession,
stoppages called or missed) and says why each unmatched true possession failed; results in
`possession/README.md`.

`score.py` in-sample (weights fitted on all 30): IDSSE 0.244, Metrica 0.311, SkillCorner 0.300,
all 30 0.288 spellF1. Before the stoppage model (median speed under 1.2 m/s): held-out mean 0.250,
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

## Stoppage edges: `hmm.decode --hold-restart`, `edge_diag.py`, `edge_post.py`

Motivation, stated plainly: this rule came from error analysis by video of BAR-ATH, the scored
match, under hmm_v9 (2026-11-16). Restarts were called live several seconds before the kick, with
the ball put down and still and a player standing over it, which the gap clue reads as
possession. Its parameters were chosen and swept on the outside providers only; nothing about
BAR-ATH is in the code. `hold_restarts` (hmm/decode.py, off by default) moves the start of the
spell after a stoppage to the kick, where a seen ball that was still (under 1 m/s for 0.5 s) is
kicked (over 5 m/s) within 10 s and before that spell ends. A spell is moved, never removed.

| | IDSSE | Metrica | SkillCorner | mean | fair | BAR-ATH public |
|---|---|---|---|---|---|---|
| v9 | 0.725 | 0.807 | 0.496 | 0.676 | 0.663 | 0.774 |
| v9 + `--hold-restart` | 0.726 | 0.807 | 0.496 | 0.6764 | 0.663 | 0.787 |

Held out it is flat in every setting tried (still 0.5-2 m/s, kick 3-5 m/s, 0.5-2 s still:
0.6761-0.6765), because the outside providers hide or freeze the ball in a stoppage and the rule
rarely fires there, so the held-out numbers are no evidence either way. The public-window gain is
one restart (1.0 s early instead of 5.4 s), well inside the bootstrap band. On the full match it
fires 7 times, moving restarts by 1.4-6.0 s; `--hold-restart` prints each firing with the
nearest Smart Tagging set piece (4 within 6 s, 3 with none within 18 s; Smart Tagging misses
about half of all set pieces). It does not reach throw-ins (the ball moves with the thrower) or
restarts with the ball unseen.

Also measured, not kept: turning an H or A spell between two decoded stoppages into D. Under
1.5 s it is flat (0.6762), from 2 s on it costs (2 s 0.6754, 3 s 0.6715). Spell errors on the
held-out decodes (edge_diag.py dumps them) are mostly 2-5 s possessions of the other team, missed
(IDSSE 116, Metrica 32, SkillCorner 293) and invented (30, 4, 428) in similar numbers, which is
team-naming quality, not a boundary bias a rule can move. Restart timing on the outside matches
is already within 0.3 s at the median.

## hmm_v9: the players-only net as a clue, `score_fuse.py`

    python external/control_features.py && python external/train_control.py --loso
    sh deep/queue.sh deep/jobs_v9.txt          # six held-out fold nets, about 3 hours on 3 cores
    python external/score_fuse.py --tag bt,b2  # variants are "control weight:net weight"

v9 is v8 with one more term in the decoder's log emissions: the per-frame p(H, A, D) of the
players-only net of `deep/`, as a log ratio against the class shares it was trained on, at a
weight of 0.3 (`hmm/net_clue.py`, `hmm.decode.NET_WEIGHT`). Nothing else changes: same control
classifier, same tables, same chain. Each match is read by nets that never saw its provider.
Folded spellF1, measured 2026-11-16 (v8 rerun the same day reproduces 0.644 / 0.816 / 0.480):

| folded spellF1 | v8 | net alone in the decoder (no control clue) | v9, one net (`bt`) | **v9, two nets averaged** | v9 - v8, paired (se, matches up) |
|---|---|---|---|---|---|
| IDSSE (7) | 0.644 | 0.692 | 0.716 | **0.728** | +0.084 (0.010, 7 of 7) |
| Metrica (3) | 0.816 | 0.799 | 0.808 | **0.807** | -0.009 (0.007, 1 of 3) |
| SkillCorner (20) | 0.480 | 0.489 | 0.496 | **0.497** | +0.016 (0.004, 16 of 20) |
| mean of the three | 0.647 | 0.660 | 0.674 | **0.677** | |

Against the labels as they are: 0.576 to 0.592. Alive/dead F1 0.946 to 0.951, team accuracy on
frames both call live 0.904 to 0.910 (IDSSE 0.900 to 0.921, Metrica 0.966 to 0.964, SkillCorner
0.845 to 0.845). With `--min-possession 1.5` (still off): 0.652 to 0.681.

**The weight was read off these matches.** Two nets, control weight 0.2:

| net weight | 0 (v8) | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| IDSSE | 0.644 | 0.670 | 0.689 | 0.705 | 0.715 | 0.721 | 0.728 | 0.727 | 0.726 |
| Metrica | 0.816 | 0.819 | 0.816 | 0.810 | 0.807 | 0.810 | 0.807 | 0.800 | 0.794 |
| SkillCorner | 0.480 | 0.489 | 0.495 | 0.495 | 0.496 | 0.497 | 0.497 | 0.490 | 0.487 |
| mean | 0.647 | 0.659 | 0.667 | 0.670 | 0.673 | 0.676 | 0.677 | 0.673 | 0.669 |

Picked for each provider on the other two, the folds choose 0.1, 0.3 and 0.3 and the provider
left out scores 0.689 / 0.807 / 0.497: a mean of **0.664**, which is the fair size. The gap to
0.677 is all IDSSE: Metrica and SkillCorner together are flat from 0.05 to 0.3, so they cannot
see that IDSSE wants more of the net.

**Where the gain is.** IDSSE's ball is knocked out to 59% seen in short random gaps, and that is
where v8 loses most there (with the true ball back v8 scores 0.752 on IDSSE; measured in another
thread the same day). The net reads all 22 players and fills those gaps. Metrica's labels and
ball come from the same events and its ball is only missing in stoppages, so there the net has
nothing to add and smooths a few short possessions away. SkillCorner gains in alive/dead (0.942
to 0.954), not in team naming.

**The two nets.** `bt` is `deep/`'s shipped net. `b2` is, for the IDSSE and Metrica folds,
`sw25` (the same net with SkillCorner's labels at a quarter weight, `--provw skillcorner=0.25`,
the one variant `deep/README.md` found level or up everywhere), and for the SkillCorner fold,
where `sw25` is `bt`, a second seed of `bt`. `b2` alone in v9 scores 0.735 / 0.804 / 0.496
(0.679); the average is kept because two nets are steadier than one (`deep/`'s seed spread is
0.013). For BAR-ATH the decoder averages `deep/models/bt.pt` and `deep/models/sw25.pt`, both
trained on all 30 matches.

Tried on the way and not kept (one net, mean of the three):

| | mean | note |
|---|---|---|
| the net's outputs as 45 columns of the control classifier (`net_features.py`, `train_control.py --net bt`): p(H, A, D), the pointer's lean, pointer and ball guess against the visible ball, windows | 0.658 (0.659 / 0.827 / 0.487) | up on v8 everywhere, but well under the sum in the decoder; held-out log loss on SkillCorner worsens 0.76 to 0.87 |
| that classifier and the net in the decoder together | 0.662 to 0.668 | no better than v8's classifier with the net |
| a different net weight where the ball is not seen (0.5 to 0.8 against 0.1 to 0.3) | 0.667 to 0.672 | flat |
| the net's alive/dead part at half or 1.5 times its team part | 0.672 | flat |
| without the width clue, or the hand-built clues at 0.15 | 0.673, 0.671 | flat |
| control weight 0.1 to 0.3 beside the net | 0.669 to 0.674 | flat |

**What ships is a near neighbour of what was measured.** The fold nets behind the tables are
trained for 1,500 steps with a provider held out (`deep/jobs_v9.txt`); `bt.pt` and `sw25.pt` are
trained for 2,000 steps on all 30 matches. On the SkillCorner fold `b2` is a second seed of
`bt`, not the `sw25` recipe, since `sw25` only differs where SkillCorner is in training. And
`hmm.net_clue.NET_PRIOR` counts the three providers equally, while `sw25` trains with
SkillCorner's labels at a quarter weight, so its implied class shares differ a little (D 0.38
against 0.35). No provider mean here has a bootstrap interval: with 3, 7 and 20 matches, the
paired standard errors in the first table are what there is, and by them Metrica's drop is noise
and SkillCorner's gain is small but consistent.

**On BAR-ATH (2026-11-16, check only; nothing was chosen on it).** `python -m hmm.decode` and
`python -m hmm.decode --no-net` on the tactical tracking, with `deep/run.py` (`bt` alone) beside
them:

| full match, no labels read | `hmm_v8` | `hmm_v9` | `deep` (`bt`) |
|---|---|---|---|
| possessions, half 1 / half 2 | 127 / 137 | 117 / 124 | 125 / 147 |
| mean possession | 13.6 / 12.0 s | 14.9 / 13.3 s | 13.9 / 11.4 s |
| median possession | 8.0 / 7.8 s | 8.4 / 9.1 s | 7.5 / 7.0 s |
| possessions under 2 s | 10% / 10% | 6% / 8% | |
| dead share of graded frames | 46% / 48% | 45% / 48% | 45% / 47% |
| home's share of live frames | 64% / 58% | 66% / 59% | 65% / 58% |
| stoppages | 110 | 102 | |
| a decoded restart within 2 s of the 58 tagged set pieces | 54 | 53 | |

v9 has 8.7% fewer possessions than v8 (241 against 264) and the same dead share. The public
window's possessions have a mean of 14.2 s. v8 and v9 differ on 5.8% of graded frames: 6.8% where
the ball is seen (59% of frames) and 4.3% where it is not; where both call the ball live they
name different teams on 5.0%; v8 is dead where v9 is live on 1.7% and the reverse on 1.5%. v9
sits between its two parents: it differs from `deep` on 5.3% of graded frames, where v8 differs
from `deep` on 10.5%.

| public 5 minutes, `run_evaluation.py --bootstrap 1000` (17 possessions) | spellF1 (5-95%) | macroF1 (5-95%) | alive/dead | team, both alive | possessions |
|---|---|---|---|---|---|
| `hmm_v8` | 0.679 (0.48 to 0.84) | 0.861 (0.62 to 0.90) | 0.962 | 0.916 | 16 |
| `hmm_v9` | 0.774 (0.58 to 0.90) | 0.904 (0.64 to 0.95) | 0.972 | 0.943 | 14 |
| `deep` (`bt`) | 0.745 (0.51 to 0.89) | 0.900 (0.68 to 0.97) | 0.960 | 0.977 | 12 |

The intervals overlap almost entirely, so the window does not say v9 is better; it says nothing
has gone wrong. `hmm_v9.csv` passes `evaluation/validate_submission.py`, and two runs on the same
machine (Intel Xeon with AVX-512, torch 2.15.0, one thread) give the same file, sha256
`cf1a838d...2ce02d31`. It has not been regenerated on a second CPU. `python -m hmm.decode` needs
torch (`requirements-deep.txt`); `--no-net` gives v8 without it. The outside sets lose the ball at
random and BAR-ATH loses it around restarts in long gaps (above), so the size of IDSSE's gain
should not be expected here; the direction on the public window is the same.

## hmm_v8: the control classifier retrained, `score_control.py`

    python external/control_features.py && python external/train_control.py --loso
    python external/score_control.py

The number to compare a change on is the **folded** column: spellF1 against the labels folded
towards the challenge's controlled-possession rule (`external/control.py`), each provider scored
by a classifier trained on the other two, ball at `--seen 0.59`. Measured 2026-11-14:

| folded spellF1 | hmm_v6 | hmm_v7 (weight 0.15) | hmm_v8 (weight 0.2) | v8 - v7 | v8 with `--min-possession 1.5` |
|---|---|---|---|---|---|
| IDSSE (7) | 0.510 | 0.609 | 0.644 | +0.035 | 0.650 |
| Metrica (3) | 0.764 | 0.798 | 0.816 | +0.018 | 0.819 |
| SkillCorner (20) | 0.403 | 0.428 | 0.480 | +0.052 | 0.486 |
| mean of the three | 0.559 | 0.612 | 0.647 | +0.035 | 0.652 |

v8 is up on v6 on every match (7 of 7, 3 of 3, 20 of 20). Against the labels as they are it
scores 0.543 / 0.765 / 0.420 (v7: 0.523 / 0.769 / 0.376).

**The weight was read off these matches.** Chosen for each provider on the other two (0.15 / 0.2 /
0.3), the folds pick 0.15, 0.3 and 0.3, and the provider left out then scores 0.636 / 0.817 /
0.476: a mean of **0.643**, which is the fair size. (With the fold length chosen the same way:
0.646.) Every setting in this grid has a mean between 0.645 and 0.655 and no provider below v7:

| mean (IDSSE / Metrica / SkillCorner) | weight 0.15 | 0.2 | 0.3 |
|---|---|---|---|
| no fold | 0.645 (0.636 / 0.819 / 0.480) | **0.647** (0.644 / 0.816 / 0.480) | 0.649 (0.653 / 0.817 / 0.476) |
| fold under 1 s | 0.646 (0.640 / 0.817 / 0.481) | 0.647 (0.647 / 0.812 / 0.481) | 0.648 (0.655 / 0.813 / 0.477) |
| fold under 1.5 s | 0.650 (0.641 / 0.823 / 0.485) | 0.652 (0.650 / 0.819 / 0.486) | 0.655 (0.659 / 0.822 / 0.483) |
| fold under 2 s | 0.649 (0.634 / 0.823 / 0.490) | 0.651 (0.641 / 0.820 / 0.492) | 0.652 (0.649 / 0.819 / 0.489) |

The same classifier built by the scratch scripts that came before this code scored 0.644 where
this code scores 0.645 (weight 0.15, no fold), and a build with three extra signals (below) scored
0.640, so differences under 0.005 here are noise.

What changed, one step at a time (mean of the three, weight 0.15 unless said):

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| v7 | 0.609 | 0.798 | 0.428 | 0.612 |
| without the ball-alone columns (`--drop seen,ball_speed,bspeed,ball_dv`) | 0.601 | 0.794 | 0.458 | 0.617 |
| ... providers weighted equally, regularised, no early stopping | 0.610 | 0.796 | 0.459 | 0.621 |
| the team-shape features alone, same training | 0.614 | 0.807 | 0.481 | 0.634 |
| both feature sets (the v8 classifier) | 0.636 | 0.819 | 0.480 | 0.645 |
| ... weight 0.2 (v8) | 0.644 | 0.816 | 0.480 | 0.647 |
| ... with brief possessions folded at 1.5 s (not the default) | 0.650 | 0.819 | 0.486 | 0.652 |

The first three rows after v7 were measured with scratch scripts before the code in this
repository was written, and differ from it in small ways (the ball outside the pitch was not
masked for the team-shape features).

- **The ball on its own is left out** (`hmm.control.BALL_ALONE`, and seq's `k_` group). IDSSE
  freezes the ball in a stoppage and Metrica hides it, and v7's classifier had learned that: held
  out, it called 16% of graded SkillCorner frames dead against a true 29%, with a log loss of
  1.08. Now 0.76.
- **The team-shape features of `seq/features.py`** (`p_` and `b_` groups) sit next to the control
  features in one classifier (`hmm.control.joint_features`). They read all 22 players, and carry
  most of the SkillCorner gain; the two sets add up on IDSSE.
- **Training:** the three providers count equally, at least 200 rows a leaf, L2 1.0, 300 rounds
  with no early stopping (v7 stopped on a random 10% of rows, which are neighbours of the
  training rows).
- **Folding brief possessions is built but off** (`hmm.decode.fold_brief_possessions`,
  `--min-possession`): a decoded possession under 1.5 s with the other team on both sides goes
  back to that team. It gains 0.005 here, but the folded labels are made by a rule of the same
  shape (`external/control.py`), so part of that is the yardstick agreeing with itself: against the
  labels as they are, the fold costs Metrica 0.014 and IDSSE 0.016. And the challenge's ground
  truth keeps such possessions: the public window has two among its 17, 0.76 s and 1.32 s long,
  each between two spells of the other team. That was looked at once to check the assumption, not
  to choose a number.

Tried and not kept, each scored the same way:

| | mean | note |
|---|---|---|
| a touch-balance signal, the ball's direction of travel, 20 s windows | 0.631 to 0.647 at 0.15 to 0.2, no fold | no better than without; the ball's direction costs SkillCorner 0.025 |
| separate weights for the classifier's alive/dead and team parts | 0.642 to 0.653 | flat around equal weights |
| clipping the classifier's log ratio at 1 to 3 | 0.609 to 0.612 on v7 | nothing |
| the chain's mean possession 20 s, or 60% of possessions ending dead | +0.001 | nothing |
| a second classifier over the first one's posteriors through time, trained on cross-provider posteriors | 0.640 | better team accuracy, worse alive/dead on SkillCorner |
| training IDSSE and Metrica on broadcast-like copies (branch `claude/project-thread-hqpy4o`) | +0.004 | noise-sized once the ball-alone columns are out |
| per-match prior or temperature correction (branch `claude/project-thread-kvq2x5`) | +0.005 at best on v7 | the decoder adds a likelihood ratio, which a prior shift does not change |

**What is left** is naming the team in open play on SkillCorner: its team accuracy is 0.845 on
frames both call live (IDSSE 0.900, Metrica 0.967), and none of the above moved it by more than
0.005.

**Not checked on BAR-ATH.** None of this was run on our match: the gated data was not on the
machine that did this work. `python -m hmm.decode` now needs the retrained
`hmm/models/control_state.npz` and writes `hmm_v8.csv`. Before submitting it, compare it with
`hmm_v7.csv` on the public window and by eye, as v7 was. The team-shape features depend on
tracking quality the outside sets do not show (`p_tracked`, the defensive lines, the 6-player
floor), and how they behave on BAR-ATH's fragmented slots is unchecked, as it was for `seq/`.

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
fragmented slots is unchecked. `uv.lock` includes the `seq` group (`uv sync --group seq`).
