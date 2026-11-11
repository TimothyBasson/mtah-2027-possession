# Out-of-sample matches

The public 5 minutes may not be used to tune (RULES.md §2), so we validate on **other games**, which
RULES.md allows in any amount. Ten labelled matches, in the same arrays `possession/load.py` gives
for ours:

```bash
uv run python external/build.py            # -> data/external/*.npz (gitignored, ~1 min/match)
uv run python external/score.py --seen 0.59 [param=value ...]
uv run python external/score_hmm.py --seen 0.59   # the HMM decoder in hmm/, and smooth50
```

| set | matches | tracking | label source |
|---|---|---|---|
| `idsse_*` | 7 (2 Bundesliga, 5 2. Bundesliga, 2022-23) | TRACAB optical, 25 fps, ball ~always seen | DFL's per-frame `BallStatus` / `BallPossession` |
| `metrica_*` | 3 anonymised sample games | Metrica, 25 fps, ball seen ~60% (like ours) | rebuilt from events, see `build.py` |

Where the raw files come from (put them under `data/external/raw/`):

- IDSSE: [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0,
  Bassek et al., *Scientific Data* 12, 195 (2025). The figshare originals sit behind a bot check.
- Metrica: `git clone https://github.com/metrica-sports/sample-data` into `raw/metrica/`.

## What the labels are, and are not

Neither label source uses the challenge's own "controlled possession" rule, so absolute scores
here are not leaderboard predictions. Use them to compare two versions of the model.

- **DFL `BallPossession`** is set by data operators and likely flips on the first touch more than
  our rule does.
- **Metrica events:** dead from a ball-out, foul or shot-out until the next on-ball action. While
  alive, possession belongs to the team of the last pass, carry, shot, recovery or set piece.
  Challenges and lost balls don't change it.

**DFL's spells are short.** A match has about 510, with a median of 3 s. Our 5 public minutes
have 17 possessions, roughly 360 over a full match, so DFL counts some touches we would not.
Metrica has about 315, which is closer. Expect a low spellF1 on IDSSE even when the model is right
by the organisers' rule. That set is better for alive/dead and team accuracy.

## Baseline: v3 model, `--seen 0.59`, no tags

| | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE mean (7) | 0.369 | 0.803 | 0.881 | 0.873 |
| Metrica mean (3) | 0.576 | 0.849 | 0.918 | 0.920 |
| all 10 | 0.431 | 0.817 | 0.892 | 0.887 |

For comparison, public spellF1 is 0.594. With the ball always seen, IDSSE rises to 0.458 spellF1
and 0.975 alive/dead.

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

Use `--seen` on IDSSE. With the ball always visible, the model's hardest problem disappears.
Scores also get no help from set-piece tags, because these matches have no Smart Tagging.
