# alldata/holder: a per-player "who has the ball" branch on the all-data model (measured, not adopted)

Question: `alldata/learned/` sees the players only through summaries of each team's three
nearest to the ball. Does a network that looks at the players one by one, with inputs relative
to the ball, and names the holder (where the recent literature sits) raise spellF1?

Answer: no. Two seeds, held out by provider, BAR-ATH-like balls, SkillCorner at a quarter of the
training windows, same decode as `alldata/learned/score.py`:

| | IDSSE | Metrica | SkillCorner | mean, labels as built | mean, folded labels |
|---|---|---|---|---|---|
| `fit.py`, seed 0 (`w1`) | 0.643 / 0.723 | 0.767 / 0.808 | 0.405 / 0.522 | 0.605 | 0.684 |
| `fit.py`, seed 1 (`w1s1`) | 0.650 / 0.725 | 0.778 / 0.814 | 0.404 / 0.512 | 0.611 | 0.684 |
| `hfit.py`, seed 0 (`k1`) | 0.636 / 0.730 | 0.768 / 0.799 | 0.408 / 0.516 | 0.604 | 0.682 |
| `hfit.py`, seed 1 (`k1s1`) | 0.645 / 0.730 | 0.773 / 0.806 | 0.408 / 0.520 | 0.609 | 0.685 |
| `fit.py`, seeds averaged (`w1e`) | 0.648 / 0.727 | 0.781 / 0.818 | 0.409 / 0.519 | 0.613 | 0.688 |
| `hfit.py`, seeds averaged (`k1e`) | 0.650 / 0.739 | 0.786 / 0.813 | 0.409 / 0.519 | 0.615 | 0.690 |

(as built / folded per provider.) The difference of the averaged rows is +0.001 / +0.002 with a
paired standard error of 0.002 / 0.003 over the 30 matches, 15 and 14 of them up; seed to seed
the same model moves by more. Team accuracy 0.897 against 0.894, alive/dead F1 0.961 both. The
baseline rows reproduce `alldata/learned/README.md` (0.613 / 0.688 before adaptation).

The adapted runs (`--pseudo`, the step from 0.613 / 0.688 to the shipped 0.616 / 0.696) were not
trained for the new network: it is level before adaptation, the step is the same procedure for
both, and it would have cost two more hours of four CPUs. So no BAR-ATH file was made and the
shipped file is unchanged.

## What was built (`hfit.py`)

- **Tokens.** Each team's five nearest players to the ball, per row (0.2 s). Inputs per player:
  offset to the ball, distance and its rank (overall and in the team), own velocity, velocity
  against the ball's, closing speed, where the ball's path passes the player and when, the
  player's own distance to the ball at 0.2, 0.4 and 1 s either side and the nearest approach,
  team, position, and whether the ball is seen.
- **Pointer.** The ten tokens attend to each other (one layer) and each gets a score: a
  distribution over players, "who has the ball".
- **Two lessons for the pointer.** On live frames its mass on the labelled team is pushed up.
  And where the ball as recorded lies within 2.5 m of a player of the labelled team, that
  player is the target, while the network is shown the BAR-ATH-like ball: the sharp ball
  teaches how to read the smoothed one (about a third of rows have such a target).
- **Into the trunk.** Each team's pointer-weighted player state and pointer mass are added to
  the frame rows; trunk, heads, loss, sampling and outputs are `fit.py`'s.

This differs from the earlier per-player attempts (`team/`, players only, a second stage on a
finished model): the players are read against the ball, the holder is supervised from the raw
ball, and the branch is trained inside the model.

## Why it does not help (`pointer.py`, seed 0, held-out matches, BAR-ATH-like ball)

| | holder named: pointer | holder named: nearest player | team on live frames: pointer | nearest player's team | network output |
|---|---|---|---|---|---|
| IDSSE | 0.927 | 0.899 | 0.903 | 0.755 | 0.901 |
| Metrica | 0.943 | 0.908 | 0.896 | 0.755 | 0.899 |
| SkillCorner | 0.867 | 0.860 | 0.821 | 0.722 | 0.818 |

The pointer works: it names the raw-ball holder more often than "nearest to the smoothed ball"
does. But the team it points at is right exactly as often as the network's own output, with or
without the branch (0.894 without). The frame rows (three nearest per team, touches read off
the ball's velocity, the players-only net) already carry what the per-player view has to say
about which team has the ball; what is still missed is not in the players' geometry around a
ball smoothed at 0.7 Hz. Same conclusion as the turnover detector and the team-naming step.

## Not done

- A build that carried every tracked player along a path of their own in time (dilated
  convolutions per player, attention across all players) was written first; it cost 8 s a
  step on one CPU against 0.7 s for `fit.py` and 1.5 s for this one, and was dropped unscored.
- A CRF over holder paths (PathCRF) in place of the Viterbi decode over teams.
- The adapted runs and a BAR-ATH file (see above).

## Reproduce

```bash
python alldata/learned/build.py --no-tim   # feature rows (needs data/cache/deep/bt)
python alldata/holder/pcache.py            # players' paths
sh alldata/holder/queue.sh                 # w1, k1, w1s1, k1s1 (and the adapted runs, not run here); about 45 min per k tag on 4 CPUs
python alldata/learned/score.py --tags w1e k1e --key s
python alldata/holder/pointer.py --tag k1
```
