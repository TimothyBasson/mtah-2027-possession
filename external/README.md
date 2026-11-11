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

## Baseline: `possession/` v0 (players only: more spread-out team, keepers left out, has it; walking = D; runs under 2 s (H/A) or 3 s (D) absorbed)

| | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE mean (7) | 0.200 | 0.677 | 0.844 | 0.761 |
| Metrica mean (3) | 0.272 | 0.678 | 0.849 | 0.792 |
| all 10 | 0.222 | 0.677 | 0.846 | 0.771 |

Before the short runs were absorbed (`min_team=0 min_dead=0`): 0.187 / 0.212 / 0.194 spellF1.

It never reads the ball, so `--seen` doesn't change these. The ball-based nearest-team v0 before it
scored 0.057 spellF1 on all 10 at `--seen 0.59`.

The old v3 model these numbers used to describe (0.431 across all 10, ball-based) was never committed
and is lost. It is still the best single-model score seen here: this v0 is a players-only lens
meant to be combined with `hmm/`, not a replacement for it.

Use `--seen` on IDSSE. With the ball always visible, the model's hardest problem disappears.
But `--seen` drops the ball at random, live or dead alike, and our match does not lose it that way.
With the ball missing, BAR-ATH's players move at a median 1.0 m/s against 1.9 m/s with it seen,
which is the dead-ball pattern. Metrica shows the same pattern: there the ball is missing on 92% of
dead frames and 12% of live ones. So IDSSE under `--seen` punishes "missing means dead" more than
our match would.
Scores also get no help from set-piece tags, because these matches have no Smart Tagging.
