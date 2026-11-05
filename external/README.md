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

## Baseline: `possession/` v1, `--seen 0.59`, no tags

| | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE mean (7) | 0.269 | 0.723 | 0.871 | 0.794 |
| Metrica mean (3) | 0.438 | 0.746 | 0.876 | 0.857 |
| all 10 | 0.320 | 0.730 | 0.873 | 0.813 |

The old v3 model these numbers used to describe (0.431 across all 10) was never committed and is lost.

Use `--seen` on IDSSE. With the ball always visible, the model's hardest problem disappears.
Scores also get no help from set-piece tags, because these matches have no Smart Tagging.
