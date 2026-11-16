# Team reactions as evidence for brief possessions

Failure mode 1 of the final model's review: a brief possession is given to the other team
(a player wins the ball under pressure, keeps it 2 to 5 s and loses it; worth up to 0.149 on
control-rule labels). The idea tested here: when a team wins the ball, both teams react as a
whole within a second or two (the winner opens and moves up, the loser turns, drops and closes),
and since the match is labelled offline, that reaction just after each moment can be read as
evidence of who really had the ball.

**It does not work, and nothing shipped changes.** `alldata/learned/` and its files are untouched.
The teams do not react fast enough: the reaction takes about four seconds to show, by which time
a brief possession is over, and inside the failure slice the teams' movement backs the wrong team.

All numbers are held out by provider on the all-data model's saved probabilities (`pw1e`:
BAR-ATH-like balls, two nets averaged, both heads averaged and decoded as `run.py` does: cost 4,
1 s / 1 s). Held-out probabilities of the with-Tim submission were never saved, so `pw1e`
(0.609 as built / 0.692 control-rule under this decode) is the base, as in `alldata/shortdec/`.
Columns are IDSSE / Metrica / SkillCorner (/ mean of providers).

## 1. The premise: how the teams move after a turnover (`premise.py`)

At every true turnover in open play (control-rule labels), the movement of the two teams over the
window that follows, counted in the winner's direction of attack. `both` is the sum of the two
teams' mean advance in m/s (positive: both move the winner's way); `open` is the winner's change
of spread minus the loser's. In brackets, how often the number is positive (0.50 = nothing to see).

| possession lasts | n | both, 1 s before | both, next 1 s | both, next 2 s | both, next 4 s | open, next 2 s | open, next 4 s |
|---|---|---|---|---|---|---|---|
| IDSSE, 2 to 5 s | 528 | -2.22 (0.23) | -1.12 (0.37) | -0.37 (0.45) | +0.30 (0.54) | +0.12 (0.58) | +0.12 (0.59) |
| IDSSE, 5 to 10 s | 361 | -2.25 (0.23) | -1.04 (0.34) | -0.15 (0.48) | +1.12 (0.63) | +0.21 (0.63) | +0.31 (0.71) |
| IDSSE, over 10 s | 450 | -2.90 (0.15) | -1.85 (0.25) | -1.08 (0.35) | -0.01 (0.48) | +0.27 (0.68) | +0.40 (0.78) |
| Metrica, 2 to 5 s | 166 | -2.36 (0.20) | -1.28 (0.28) | -0.48 (0.46) | +0.36 (0.55) | +0.17 (0.61) | +0.18 (0.61) |
| Metrica, 5 to 10 s | 130 | -2.19 (0.23) | -1.52 (0.32) | -0.90 (0.42) | +0.23 (0.49) | +0.31 (0.71) | +0.42 (0.78) |
| Metrica, over 10 s | 255 | -2.32 (0.21) | -1.51 (0.29) | -0.88 (0.36) | +0.10 (0.52) | +0.33 (0.71) | +0.48 (0.78) |
| SkillCorner, 2 to 5 s | 845 | -0.87 (0.38) | -0.29 (0.46) | +0.03 (0.51) | +0.36 (0.55) | +0.17 (0.59) | +0.14 (0.56) |
| SkillCorner, 5 to 10 s | 798 | -1.09 (0.34) | -0.36 (0.44) | +0.08 (0.49) | +0.77 (0.57) | +0.29 (0.68) | +0.32 (0.70) |
| SkillCorner, over 10 s | 1262 | -1.42 (0.30) | -0.87 (0.38) | -0.56 (0.42) | -0.04 (0.47) | +0.34 (0.70) | +0.36 (0.73) |

Under 2 s there are only 17 to 58 turnovers per provider on these labels (full table: run the
script).

- Before a turnover both teams are moving the loser's way at about 2 m/s (it was attacking). Two
  seconds after, they are on average **still** moving the loser's way; the direction turns by
  four seconds. For a possession of 2 to 5 s the next-2 s movement is a coin flip (0.45 to 0.51).
- Opening and closing is the better sign, but weak at this range: 0.58 to 0.61 over the next 2 s
  for a 2 to 5 s possession. It grows with the length of the possession (0.73 to 0.78 over 4 s
  when it lasts over 10 s): the reaction belongs to possessions that last.

## 2. Reactions alone, inside the failure slice (`measure.py`)

A gradient-boosted classifier reads only the reaction rows (`react.py`: per team, in windows of
2 s and 1 s back and 1, 2 and 4 s ahead: advance, change of spread, length and width, share of
players advancing, the back four, the front three, closing on the ball; plus each team's shape
and players behind the ball) and names the team in possession. Fitted on two providers' live
frames, applied to the third. Share of frames with the right team named:

| | all live play | brief possessions the base found | brief possessions the base gave to the other team | (possessions) |
|---|---|---|---|---|
| reactions, fitted on all live frames | 0.881 / 0.881 / 0.834 | 0.642 / 0.648 / 0.814 | **0.161 / 0.165 / 0.117** | 191 / 39 / 630 |
| same without the columns that read the ball | 0.873 / 0.870 / 0.831 | 0.622 / 0.578 / 0.792 | 0.184 / 0.141 / 0.123 | |
| base model's own probabilities | 0.918 / 0.927 / 0.835 | 0.932 / 0.940 / 0.904 | 0.089 / 0.126 / 0.071 | |
| reactions, fitted on brief possessions only | 0.445 / 0.420 / 0.590 | 0.855 / 0.851 / 0.827 | 0.684 / 0.862 / 0.529 | |

"Brief" is a true control-rule possession under 5 s.

- The teams' movement is a good reader of who has the ball in general (0.88, players only, close
  to the base's 0.92), and **in the failure slice it names the wrong team 82 to 88% of the time**,
  as the base does. In a brief possession the two teams go on behaving as if the other team had
  the ball. The reactions and the base model are wrong together.
- Fitted on brief possessions only, the classifier is right inside them, but it has learnt the
  opposite rule ("the team that does not look in possession"): on all live play it is below a
  coin flip (0.42 to 0.59). That is no help without first knowing that a moment is inside a brief
  possession, which is the turnover the base does not see.

## 3. As a correction to the base model (`resid.py`, `stack.py`)

`resid.py`: on live rows, the base's log-odds of Home against Away are the starting score of a
gradient-boosted model (100 trees) that reads the reaction rows; its output is added to the
base's log-odds at a scale, alive/dead untouched, then decoded like the submission. Fitted on
the other providers' held-out matches. Two seeds, their average shown (seeds agree within
0.002). "To other team" is the share of true brief possessions (control-rule, under 5 s) that
the decode gives mostly to the other team: the failure slice.

| correction | labels as built | control-rule labels | brief: to other team |
|---|---|---|---|
| base (`pw1e`) | 0.632 / 0.769 / 0.426 / **0.609** | 0.759 / 0.836 / 0.482 / **0.692** | 0.22 / 0.15 / 0.37 |
| reactions, fitted on all three providers, x1 (one seed) | 0.515 / 0.739 / 0.418 / 0.558 | 0.633 / 0.802 / 0.470 / 0.635 | 0.34 / 0.17 / 0.37 |
| reactions, fitted without SkillCorner, x0.5 | 0.631 / 0.775 / 0.422 / 0.609 | 0.756 / 0.830 / 0.476 / 0.687 | 0.22 / 0.14 / 0.37 |
| same, x1 | 0.627 / 0.775 / 0.419 / 0.607 | 0.745 / 0.823 / 0.470 / 0.679 | 0.22 / 0.14 / 0.37 |
| reactions, brief possessions weighted 5, x0.5 | 0.645 / 0.777 / 0.419 / 0.613 | 0.762 / 0.825 / 0.468 / 0.685 | 0.20 / 0.13 / 0.36 |
| same, x1 | 0.656 / 0.771 / 0.410 / 0.612 | 0.752 / 0.809 / 0.455 / 0.672 | 0.17 / 0.11 / 0.37 |
| **control: no reaction inputs**, brief weighted 5, x0.5 | 0.652 / 0.774 / 0.417 / 0.614 | 0.764 / 0.822 / 0.465 / 0.684 | 0.19 / 0.13 / 0.36 |
| same, x1 | 0.663 / 0.761 / 0.404 / 0.609 | 0.741 / 0.772 / 0.441 / 0.652 | 0.16 / 0.08 / 0.34 |
| reactions and the base's probabilities, brief weighted 5, x0.5 (one seed) | 0.653 / 0.778 / 0.417 / 0.616 | 0.764 / 0.825 / 0.466 / 0.685 | 0.19 / 0.13 / 0.36 |
| reactions, brief weighted 5, taught on control-rule labels only, x0.5 (one seed) | 0.637 / 0.777 / 0.420 / 0.612 | 0.762 / 0.829 / 0.471 / 0.687 | 0.21 / 0.14 / 0.37 |

- **Plain, the reactions add nothing**: 0.609 / 0.687 against 0.609 / 0.692, and the failure
  slice does not move (IDSSE 0.22, Metrica 0.15 to 0.14).
- Fitted with SkillCorner's matches the correction drags IDSSE down by 0.12: it learns
  SkillCorner's labelling, the known noisy one.
- **Weighting brief possessions does cut the slice** (IDSSE 0.22 to 0.17, Metrica 0.15 to 0.11)
  and lifts IDSSE on labels as built by 0.013 to 0.024, **but the control without any reaction
  input does the same or more** (IDSSE 0.16, Metrica 0.08). The effect is the weighting pushing
  the base's own probabilities to cut finer, not the reactions; and it is the trade already seen
  in `alldata/shortdec/` and `deep/short/`: up on labels as built, down on control-rule labels
  (0.692 to 0.685 at x0.5, 0.672 at x1), with Metrica and SkillCorner down under both labellings
  at x1. The bar (control-rule labels up, no provider down) is not met by any row.
- `stack.py`, a free-form corrector that outputs new H/A/D probabilities from the base's
  probabilities with or without the reaction rows, is worse than the base either way (0.527 /
  0.629 base only over two seeds, 0.523 / 0.627 with reactions on the one seed run): boosted
  probabilities are smoother than the net's and the decode then merges more.

## BAR-ATH public 5 minutes (`barath.py`; a check: 17 possessions)

The correction fitted on IDSSE's and Metrica's held-out matches, applied to the submission's own
probabilities on BAR-ATH (`tpsN + tpsNs1`). Possessions drawn per half in brackets.

| | x0.5 | x1 |
|---|---|---|
| submission | 0.613, 10 of 17 found (131, 150) | |
| reactions | 0.613, 10 of 17 (131, 154) | 0.613, 10 of 17 (133, 157) |
| reactions, brief weighted 5 | 0.613, 10 of 17 (133, 165) | 0.659, 12 of 17 (138, 177) |
| control without reactions, brief weighted 5 | 0.600, 10 of 17 (133, 174) | 0.595, 12 of 17 (173, 212) |

The one row above the submission finds two more of the 17 possessions and draws 27 more in the
second half; held out it costs 0.020 on control-rule labels and is level with its no-reaction
control on labels as built. Two possessions of 17 is inside the window's noise (the same window
has moved 0.61 to 0.75 between files that score alike held out).

## Reading

The collective reaction is real but slow, and it belongs to possessions that last. A team that
wins the ball for 2 to 5 s under pressure in its own third does not get its team-mates to open
up, nor the opponents to retreat, before the ball is lost again; for the 22 players the other
team "has" the ball throughout. This is the same wall as the players-only work
(`turnover/README.md`: short turnovers are mostly not in the players' movement), now measured at
the level of the whole team and with the future allowed. Evidence for these possessions has to
come from the ball and the players at the ball, not from the teams.

## Files

| file | what it does |
|---|---|
| `react.py` | the reaction rows per match, `data/cache/alldata/react/<match>.npz` (25 s for the 30 matches) |
| `premise.py` | section 1 |
| `measure.py` | section 2 (`--no-ball`, `--brief-only`) |
| `resid.py` | section 3 (`--no-sc`, `--weight-brief`, `--with-base [only]`, `--target`, `--scales`) |
| `stack.py` | the free-form corrector; also the report line shared by `resid.py` |
| `barath.py` | the BAR-ATH check (run `alldata/learned/run.py` first) |
| `rdata.py`, `common.py` | loading; `common.py` is `alldata/shortdec/common.py` (fast spellF1 equal to the organisers') |

Setup in a fresh container: `uv sync --group deep --group seq`, `uv pip install numba
scikit-learn lightgbm`, `data/external` and `data/tactical` in place, `pw1e` in
`data/cache/alldata/runs/`. Each script takes 1 to 6 minutes on 4 CPUs.
