# Out-of-sample matches

The public 5 minutes may not be used to tune (RULES.md §2), so we validate on **other games**, which
RULES.md allows in any amount. Thirty labelled matches, in the same arrays `possession/load.py` gives
for ours:

```bash
uv run python external/build.py            # -> data/external/*.npz (gitignored, ~1 min/match)
uv run python external/score.py --seen 0.59 [param=value ...]
uv run python external/score_hmm.py --seen 0.59   # the HMM decoder in hmm/, and smooth50 (IDSSE + Metrica only)
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

Each provider is scored with the tables counted on the other provider.

| spellF1 | smooth50 | hmm_v2 | hmm_v3 | hmm_v4 | hmm_v5 |
|---|---|---|---|---|---|
| IDSSE mean (7) | 0.227 | 0.183 | 0.223 | 0.354 | 0.418 |
| Metrica mean (3) | 0.426 | 0.390 | 0.443 | 0.650 | 0.742 |
| all 10 | 0.287 | 0.245 | 0.289 | 0.443 | 0.516 |

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

**Where both still go wrong: the team either side of a stoppage.** The truth has the same team
before and after 27% of its stoppages (IDSSE 29%, Metrica 21%); `hmm_v5` and `hmm_v5fit` have it at
49% and 47%. Few of those are stoppages that are not there (5 to 8 a match). Most are real
stoppages where the model has the wrong team on one side: 51 same-team of the 97 it shares with
the truth on IDSSE, where the truth has 28.

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
