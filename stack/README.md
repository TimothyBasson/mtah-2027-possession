# stack/: the learned and the hand-built players-only models through one decode

An experiment. Does `possession/` (hand-built signals, 0.435 held out) know anything that `deep/`
(learned, 0.529 held out with the pitch-direction fix) does not? Both read the players only.

```bash
python stack/score.py --tag v1o     # v1o: deep/'s v1 fold nets, predicted again with the direction fix
```

Held out by provider, the folds of `deep/train.py` and `external/heldout.py`. For each held-out
provider, `deep/`'s net never saw it, `possession/`'s two logistics are fitted on the other two
providers minus that fold's validation matches, and the stack's weights and every decode setting
are picked on those validation matches only. All variants share one decode (a three-state chain
with a flat switching cost, then short runs absorbed) over the same grid.

| 2026-11-13, spellF1 | IDSSE | Metrica | SkillCorner | mean | macroF1 | alive/dead | team |
|---|---|---|---|---|---|---|---|
| `deep/` alone | 0.552 | 0.636 | 0.398 | **0.529** | 0.844 | 0.936 | 0.886 |
| `possession/` alone (this decode) | 0.410 | 0.508 | 0.383 | 0.434 | 0.799 | 0.913 | 0.863 |
| deep + 0.25 x possession, log-probabilities | 0.545 | 0.619 | 0.399 | 0.521 | 0.846 | 0.937 | 0.885 |
| deep + 0.5 x possession | 0.532 | 0.614 | 0.405 | 0.517 | 0.845 | 0.938 | 0.884 |
| deep + 1 x possession | 0.522 | 0.589 | 0.405 | 0.505 | 0.842 | 0.936 | 0.882 |
| logistic stack, fitted on the validation matches | 0.550 | 0.639 | 0.401 | 0.530 | 0.846 | 0.937 | 0.887 |

**Nothing gained.** The stack scores 0.530 against 0.529 for `deep/` alone, far inside `deep/`'s
seed-to-seed spread (0.013) and under the +0.01 bar; adding `possession/`'s log-probabilities
directly makes it worse the more they count. The fitted stack says the same thing: next to
`deep/`'s log-ratio (weights 2.6 to 3.4 for the team, 2.9 to 4.6 for dead), `possession/`'s
log-odds get 0.0 to 0.4 for the team and 0.2 to 0.5 for dead. The network has already learned
what width, length, push, speed and acceleration say.

So the learned model stands alone as the players-only model, and `possession/` stays as the
readable account of what such a model looks at (its viewer page shows every signal's push).

The `deep/ alone` row reproduces `deep/decode.py --tag v1o` (0.529), and `possession/ alone`
is 0.434 here against 0.435 from `external/heldout.py`: its fit leaves out the validation
matches and the decode is `deep/`'s, not its own.
