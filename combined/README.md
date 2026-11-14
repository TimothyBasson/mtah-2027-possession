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

## `deep/` as one more clue in `hmm` v7 (2026-11-14)

`combined/deep_in_hmm.py` decodes `hmm` v7 exactly as `external/score_control.py` does and adds
one per-frame term to every state's score: a weight times (log of `deep/` v2's probability of the
state, from the fold net that never saw the match's provider, minus the log of the class share).
Nothing under `hmm/` or `deep/` changes. The bar set beforehand: the weight is picked on the other
two providers, and every provider, Metrica included, has to go up against `hmm` v7. spellF1,
"folded" as in `external/control.py`.

One weight on every frame (`python combined/deep_in_hmm.py`):

| weight | 0 (`hmm` v7) | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| IDSSE, labels as they are | 0.523 | 0.555 | 0.561 | 0.570 | 0.586 |
| Metrica | 0.769 | 0.766 | 0.761 | 0.738 | 0.702 |
| SkillCorner | 0.376 | 0.387 | 0.399 | 0.407 | 0.412 |
| IDSSE, folded | 0.609 | 0.650 | 0.665 | 0.685 | 0.701 |
| Metrica | 0.798 | 0.807 | 0.822 | 0.803 | 0.781 |
| SkillCorner | 0.428 | 0.444 | 0.459 | 0.470 | 0.473 |

Held out, the weight picked for Metrica is the largest (the other two providers keep gaining) and
Metrica drops: 0.702 against 0.769 on the labels as they are, 0.781 against 0.798 folded. This
version does not clear the bar.

A weight where the ball is usable and another where it is not (`--gated`, written usable/not).
The ball is usable in 59% of in-play frames on IDSSE and SkillCorner (dropped to match BAR-ATH)
and 63% on Metrica.

| weights | 0/0 (`hmm` v7) | 0/0.1 | 0/0.2 | 0/0.4 | 0/0.8 | 0.05/0.2 | 0.05/0.4 | 0.05/0.8 |
|---|---|---|---|---|---|---|---|---|
| IDSSE, labels as they are | 0.523 | 0.560 | 0.577 | 0.594 | 0.606 | 0.583 | 0.597 | 0.606 |
| Metrica | 0.769 | 0.773 | 0.773 | 0.769 | 0.759 | 0.766 | 0.761 | 0.750 |
| SkillCorner | 0.376 | 0.387 | 0.390 | 0.387 | 0.379 | 0.397 | 0.393 | 0.384 |
| IDSSE, folded | 0.609 | 0.648 | 0.664 | 0.677 | 0.680 | 0.676 | 0.685 | 0.687 |
| Metrica | 0.798 | 0.801 | 0.802 | 0.798 | 0.788 | 0.807 | 0.802 | 0.794 |
| SkillCorner | 0.428 | 0.440 | 0.441 | 0.435 | 0.423 | 0.452 | 0.446 | 0.433 |

Held out, weights picked on the other two providers, paired by match against `hmm` v7:

| | labels as they are | folded labels |
|---|---|---|
| IDSSE | 0.577 against 0.523 (+0.054, se 0.006, up 7/7) | 0.676 against 0.609 (+0.067, se 0.006, up 7/7) |
| Metrica | 0.750 against 0.769 (-0.018, se 0.004, up 0/3) | 0.802 against 0.798 (+0.004, se 0.006, up 2/3) |
| SkillCorner | 0.379 against 0.376 (+0.002, se 0.003, up 11/20) | 0.446 against 0.428 (+0.019, se 0.004, up 17/20) |
| provider mean | 0.569 against 0.556 | 0.641 against 0.611 |

Reading. On the folded labels, the target `hmm` v7 is built for and the closest to the challenge's
rule, the gated combination clears the bar: no provider down, mean up 0.030, though Metrica's
+0.004 is inside its noise. On the labels as they are it does not: the weight picked for Metrica
is too large and Metrica drops 0.018. The mildest gated setting, 0/0.2 (`deep/` speaks only where
there is no usable ball), is up on every provider under both labellings (+0.054 / +0.004 / +0.014
as they are, +0.055 / +0.004 / +0.013 folded), but that setting was read off this table with all
three providers in view, so it is a suggestion, not a held-out result. What is solid: `deep/`
adds a lot on IDSSE and nothing measurable either way on Metrica at small weights.

### On BAR-ATH (check only, nothing tuned on it)

`python combined/run.py` runs `hmm/decode.py` as for `hmm_v7.csv` with the gated term at 0/0.2 and
writes `evaluation/results/hmm_v7_deep.csv`. With the weight at 0 the file is byte-identical to
`hmm_v7.csv`. At 0/0.2 it agrees with `hmm_v7.csv` on 97.9% of graded frames (99.8% where the ball
is seen, 95.2% where it is missing) and has 286 possessions against 289; the two name different
teams in 14 short runs, 14 s in all, the longest 3.2 s.

Public 5 minutes, 17 possessions, so gaps under about 0.33 are noise:

| | spellF1 | macroF1 | possessions (17 true) |
|---|---|---|---|
| `deep` v2 | 0.826 | 0.918 | 14 |
| `hmm` v7 + `deep` at 0/0.2 | 0.618 | 0.812 | 16 |
| `hmm` v7 | 0.606 | 0.783 | 16 |

So at this weight the combination is `hmm` v7 with small repairs where the ball is missing. It is
a safe step up from v7 on the outside matches; it does not settle v7 against `deep`.
