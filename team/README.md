# team/: a dedicated team-naming step (2026-11-14) — measured, not adopted

Question: once the learned model (`deep/`) has found where possessions start and end, can a second
model whose only job is home-or-away raise spellF1? All numbers are held out by provider, on labels
folded to the controlled-possession rule (`external/control.py`), decode cost 4 with a 2 s minimum
possession and 3 s minimum stoppage, equal-provider mean. Players only at prediction throughout.

## 1. What perfect naming would be worth (`oracle.py`, `overseg.py`, `missed.py`)

| on `deep` v2's probabilities | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| as decoded | 0.672 | 0.723 | 0.489 | 0.628 |
| its possessions, each given the true team | 0.697 | 0.732 | 0.634 | 0.688 |
| cut as finely as the probabilities allow, true team, 2 s minimum | 0.710 | 0.737 | 0.651 | 0.700 |
| its alive/dead, the true team frame by frame | 0.891 | 0.879 | 0.871 | 0.880 |
| true boundaries, the model votes the team | 0.819 | 0.838 | 0.733 | 0.797 |

Naming the possessions stage 1 draws is worth at most +0.06, nearly all of it on SkillCorner
(its 2-5 s pieces are named right 56% of the time). The other +0.19 is turnovers that are not in
stage 1's probabilities at any cut: inside a true possession it misses, stage 1 gives the true
team 0.23 to 0.40, so it is sure and wrong, not unsure. A missed 3 s possession also merges the
two either side of it, which is why it costs so much. True team only inside true possessions of
5 s or more would give 0.748; of 2 s or more, 0.855.

## 2. A possession-level classifier on the base's output (`pieces.py`): no gain

Gradient boosting per piece: how sure and steady the base was, the neighbours, the live stretch,
both teams' centroid, width, depth and forward speed, optionally the other saved models' opinion.
Trained on the other two providers, each seen through the net that never trained on it.

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| base v2 | 0.672 | 0.723 | 0.489 | 0.628 |
| pieces renamed where the classifier says the base is wrong | 0.644 | 0.712 | 0.490 | 0.616 |
| the same with v2s1, bt, bts1 as extra evidence | 0.652 | 0.710 | 0.494 | 0.619 |
| base bt + bts1 | 0.697 | 0.739 | 0.501 | 0.646 |
| renamed, v2 + v2s1 as extra evidence | 0.672 | 0.733 | 0.502 | 0.636 |

The classifier is right about the base on 85% of pieces; "the base is always right" is right on
86%. Nothing in the base's output or the team shapes says which pieces are misnamed.

## 3. A team-only net that can place turnovers (`tdata.py`, `train.py`, `apply.py`): no gain over bt

`deep/net.py` with the ball-taught pointer, on tokens that also carry each player's own path over
the second either side (8 offsets, 0.12 s to 1 s), trained on folded labels with the loss on
"which team, on live frames" (H/A/D kept at weight 0.3). 1,500 steps, one seed, tag `t1`.
`apply.py` keeps the base's alive/dead and takes the team split from the team net (`--mix`).

| | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| v2 | 0.672 | 0.723 | 0.489 | 0.628 |
| t1 alone | 0.681 | 0.706 | 0.505 | 0.631 |
| v2 alive/dead, t1 team | 0.678 | 0.713 | 0.502 | 0.631 |
| bt (one seed) alone | 0.688 | 0.736 | 0.499 | 0.641 |
| bt + bts1 | 0.697 | 0.739 | 0.501 | 0.646 |
| bt + bts1 alive/dead, t1 team | 0.687 | 0.721 | 0.514 | 0.641 |
| bt + bts1, team split half t1 | 0.699 | 0.737 | 0.512 | 0.649 |

t1 is level with v2 and behind bt; mixed half and half into bt it adds 0.003, which is a third
net in an average, not a team step. The bar (+0.01, no provider down) is not met. Its training
loss on team falls to 0.12-0.25 while held-out team accuracy stays at 0.84-0.90, as v2's.

## Reading

Three differently built nets (v2, bt, t1) land within 0.02 of each other and miss the same kind of
possession. A second stage reading the first one's output has nothing to work with, and a second
net reading the players finds what the first found. The gain, if there is one from players alone,
has to come from the base model seeing turnovers it does not see now, not from a step after it.
Fold nets and probabilities for t1: `/mnt/project-files/deep/t1/`.
