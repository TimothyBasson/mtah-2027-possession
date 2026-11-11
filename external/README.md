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
- SkillCorner: [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT. The tracking
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
  half the dead time.

**DFL's spells are short.** A match has about 510, with a median of 3 s. Our 5 public minutes
have 17 possessions, roughly 360 over a full match, so DFL counts some touches we would not.
Metrica has about 315, and SkillCorner about 300 (median 6 s), which is closer. Expect a low
spellF1 on IDSSE even when the model is right by the organisers' rule. That set is better for alive/dead and team accuracy.

## Baseline: v3 model, `--seen 0.59`, no tags

| | spellF1 | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|
| IDSSE mean (7) | 0.369 | 0.803 | 0.881 | 0.873 |
| Metrica mean (3) | 0.576 | 0.849 | 0.918 | 0.920 |
| all 10 | 0.431 | 0.817 | 0.892 | 0.887 |

For comparison, public spellF1 is 0.594. With the ball always seen, IDSSE rises to 0.458 spellF1
and 0.975 alive/dead.

Use `--seen` on IDSSE and SkillCorner, which is measured on labelled frames. With the ball always visible, the model's hardest problem disappears.
Scores also get no help from set-piece tags, because these matches have no Smart Tagging.
