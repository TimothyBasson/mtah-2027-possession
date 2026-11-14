# combined/: a first combination of `possession/` and `hmm/` (experiment)

`possession/` reads the players only; `hmm/` reads the ball. This folder tries the three cheapest
ways to put them together. Nothing under `possession/`, `hmm/` or `seq/` is changed: both are
imported as libraries. `possession/`'s own held-out score stays 0.435.

```bash
uv run python combined/score.py              # held out by provider, ~10 minutes on 5 workers
uv run python combined/score.py --bar-ath    # combination c on BAR-ATH -> evaluation/results/combined_stack.csv
```

**Result: no combination beats the better single model by the rule (+0.01 on the three-provider
mean, no provider down).** The stack (c) has the best mean, 0.509 against `hmm_v6`'s 0.499, but
Metrica falls 0.064. The combinations help where the ball clues are weak (IDSSE, SkillCorner) and
hurt where they are strong (Metrica).

## Protocol

For each provider in turn, everything fitted or counted comes from the other two providers, each
counting once, and the held-out provider's matches are scored:

- `possession/`'s two logistic models, as `external/heldout.py` refits them (its row below
  reproduces `heldout.py` exactly)
- `hmm_v6`'s speed, flight, gap-distance and width tables, counted as `external/score_hmm.py`
  counts them
- the combination's own weight, picked by mean spellF1 on the other two providers; the stack's
  weights are fitted there

The ball is knocked out as `score_hmm.py --seen 0.59` does: IDSSE and SkillCorner (measured on
labelled frames) down to 59% seen, Metrica as recorded. `possession/` never reads the ball.

**Where this differs from `score_hmm.py`'s own protocol.** There IDSSE is scored with tables
counted on Metrica alone, Metrica with IDSSE's, and SkillCorner with the shipped constants
(IDSSE and Metrica); only the width table is counted on the two other sets. Here every counted
table comes from the two other providers, so SkillCorner's labels help count the tables for IDSSE
and Metrica. `hmm_v6` scores lower that way on IDSSE and Metrica:

| `hmm_v6` spellF1 / macroF1 / alive-dead / team | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| `score_hmm.py --seen 0.59` (rerun 2026-11-12) | 0.441 / 0.814 / 0.903 / 0.874 | 0.753 / 0.878 / 0.924 / 0.960 | 0.354 / 0.746 / 0.903 / 0.831 | 0.516 / 0.813 / 0.910 / 0.888 |
| this folder's protocol | 0.413 / 0.813 / 0.904 / 0.869 | 0.730 / 0.876 / 0.925 / 0.954 | 0.354 / 0.746 / 0.903 / 0.831 | 0.499 / 0.812 / 0.911 / 0.885 |

In both, `hmm_v6`'s emission weight (0.2), width weight (0.05), transition rates and hand-set
place table are as shipped; those were chosen on IDSSE and Metrica, so neither row is fully held
out for them.

## The combinations

- **a** `hmm_v6`'s per-frame scores plus w × `possession/`'s log P(H), log P(A), log P(D), decoded
  as `hmm/` decodes. w from 0.05, 0.1, 0.2.
- **b** `possession/`'s log-probabilities plus k × `hmm_v6`'s per-frame scores, decoded as
  `possession/` decodes (switch costs 5 and 20). k from 2, 5, 10.
- **c** two logistic stacks refitted per fold: team on (`possession/`'s log-odds of home,
  `hmm_v6`'s H minus A score), dead on (`possession/`'s log-odds of dead, `hmm_v6`'s D minus
  best-live score); then `possession/`'s decode. Nothing to pick.

Held out, spellF1 / macroF1 / alive-dead / team (2026-11-12):

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| `possession/` alone | 0.424 / 0.837 / 0.932 / 0.864 | 0.506 / 0.800 / 0.889 / 0.899 | 0.375 / 0.770 / 0.922 / 0.829 | 0.435 / 0.802 / 0.914 / 0.864 |
| `hmm_v6` alone | 0.413 / 0.813 / 0.904 / 0.869 | 0.730 / 0.876 / 0.925 / 0.954 | 0.354 / 0.746 / 0.903 / 0.831 | 0.499 / 0.812 / 0.911 / 0.885 |
| a, w picked held out (0.05, 0.2, 0.1) | 0.443 / 0.841 / 0.927 / 0.877 | 0.654 / 0.854 / 0.910 / 0.943 | 0.394 / 0.780 / 0.923 / 0.839 | 0.497 / 0.825 / 0.920 / 0.887 |
| b, k picked held out (5, 2, 5) | 0.499 / 0.861 / 0.940 / 0.886 | 0.624 / 0.836 / 0.902 / 0.931 | 0.348 / 0.779 / 0.924 / 0.835 | 0.490 / 0.825 / 0.922 / 0.884 |
| c, stack | 0.501 / 0.855 / 0.936 / 0.886 | 0.666 / 0.849 / 0.908 / 0.939 | 0.360 / 0.781 / 0.926 / 0.835 | 0.509 / 0.828 / 0.923 / 0.886 |

Every setting on the held-out provider (spellF1), to show the picks were not unlucky:

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| a, w = 0.05 | 0.443 | 0.717 | 0.385 | 0.515 |
| a, w = 0.1 | 0.453 | 0.690 | 0.394 | 0.512 |
| a, w = 0.2 | 0.455 | 0.654 | 0.398 | 0.502 |
| b, k = 2 | 0.473 | 0.624 | 0.382 | 0.493 |
| b, k = 5 | 0.499 | 0.696 | 0.348 | 0.514 |
| b, k = 10 | 0.469 | 0.689 | 0.307 | 0.488 |

## Reading it

- macroF1 and alive/dead rise on the mean with every combination (0.812 → 0.825-0.828,
  0.911 → 0.920-0.923): `possession/`'s stoppage model is the better dead-ball reader on IDSSE
  and SkillCorner.
- Metrica loses spellF1 with every combination, and more the more `possession/` counts
  (a: 0.717, 0.690, 0.654 as w grows). There `hmm_v6` is already right on short possessions that
  `possession/` cannot see, and the players-only evidence smooths them away.
- IDSSE gains most (0.413 → about 0.50 with b or c), SkillCorner a little with a and c.
- So one weight for the whole match is the wrong shape. A weight that depends on the frame is
  the next thing to try: lean on `possession/` where the ball is missing or the ball clues are
  near even, and leave `hmm/` alone where it is sure. `hmm_v6` already reads width and player
  speed, so adding `possession/` on top counts some evidence twice; dropping those two clues
  from the HMM side of the combination was not tried.

## BAR-ATH public 5 minutes (one check, nothing tuned on it)

Combination c with everything fitted and counted on all 30 outside matches:

| | macroF1 | spellF1 | possessions (17 true) |
|---|---|---|---|
| `hmm_v6` as shipped | 0.760 | 0.582 | 16 |
| combination c | 0.835 | 0.573 | 20 |
| `possession/` | 0.846 | 0.516 | 14 |

With 17 possessions these spellF1 values cannot be told apart.
