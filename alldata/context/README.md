# alldata/context: whole-match context pass (failure mode 1, brief possessions given to the wrong team)

**Status: not adopted.** The submission is unchanged.

Question: does a second pass that reads the first stage's output over a whole half, so each brief
segment is judged with minutes of context (team styles, field-position trends, who wins the ball
where), name the team of brief possessions better?

**Answer: no.** Long-range context adds nothing over a local second stage, and neither beats the
first stage by more than seed noise. Team accuracy does not move (0.898 to 0.897).

## Model (`ctx.py`)

Inputs per 5 fps frame, BAR-ATH-like ball (version 1): the first stage's held-out
log-probabilities (`pw1e`, both heads, from the fold where the match's own provider was held out),
the learned touch detector (`t1`, same folds), and the first stage's own rows (build.py). A frame
encoder and a short TCN (local), then a 3-layer transformer over the **whole half** in 2 s tokens
with absolute match-time encoding, added back at frame rate, then a short TCN. The output is a
residual on the first stage's logits (zero at the start). Trained on whole halves (random crops of
30-100%) of the other two providers' matches, SkillCorner at most a quarter of draws. Team swap
and mirror as augmentation and averaged at prediction. `--no-ctx` is the control: the same net
without the transformer. `--team-only` keeps the first stage's dead probability exactly and lets
the second stage only re-split home vs away.

## Results

Held out by provider, BAR-ATH-like ball, mean of IDSSE / Metrica / SkillCorner, fixed decode of
`alldata/learned/score.py`, then with the submission's touch evidence (`alldata/touch/tuse.py
--thr 0.3 --w 1 --hold 3 --keep 0`). As built / folded.

| | IDSSE (+ touch) | Metrica (+ touch) | SkillCorner (+ touch) | mean, plain decode | mean, + touch evidence |
|---|---|---|---|---|---|
| base `pw1e` (+ touch) | 0.666 / 0.760 | 0.794 / 0.831 | 0.417 / 0.520 | 0.616 / 0.696 | **0.626 / 0.704** |
| `cx1` full residual, seed 0 (+ touch) | 0.660 / 0.732 | 0.790 / 0.841 | 0.400 / 0.510 | 0.605 / 0.681 | 0.617 / 0.694 |
| `cx2` team-only, seed 0 (+ touch) | 0.677 / 0.756 | 0.795 / 0.843 | 0.408 / 0.519 | 0.618 / 0.698 | 0.627 / 0.706 |

IDSSE fold only (plain decode), context against its control:

| | as built | folded |
|---|---|---|
| base `pw1e` | 0.652 | 0.747 |
| `cx2` seed 0 / seed 1 | 0.660 / 0.657 | 0.749 / 0.745 |
| `nc2` no-context control | 0.660 | 0.744 |
| `cx1` / `nc1` (full residual) | 0.646 / 0.646 | 0.715 / 0.712 |

Readings:
- The full-residual version learns the other providers' dead-ball conventions: IDSSE alive/dead
  F1 0.988 to 0.932, folded IDSSE -0.033. Keeping the dead call from the first stage removes that.
- With the dead call fixed the second stage is level with the base: +0.001 / +0.002 with touch
  evidence, SkillCorner as built 0.417 to 0.408, team accuracy 0.898 to 0.897. Far from +0.01.
- The whole-half transformer and the no-context control score the same on every comparison made,
  so long-range context carries no usable signal for who owns a brief possession beyond what the
  first stage's own 60 s window already sees. This matches the earlier findings that the families
  miss the same turnovers and that which brief touches a provider labels does not transfer.
- No BAR-ATH file was built: nothing here clears the bar, and a version on the submission's
  with-Tim base would need its held-out predictions, which were never saved.

## Commands

    python alldata/learned/build.py --no-tim --versions 3     # first-stage rows (needs data/external, deep bt runs)
    python alldata/touch/tfeats.py                            # touch rows, for tuse.py's scoring
    # runs pw1e (/mnt/project-files/alldata/learned/runs/) and t1 (/mnt/project-files/alldata/touch/runs/)
    # go to data/cache/alldata/runs/ and data/cache/alldata/touch/runs/
    alldata/context/queue.sh alldata/context/jobs2.txt        # four fits at a time, ~40 min each at 500 steps
    python alldata/learned/ens.py cx2e cx2 cx2s1
    python alldata/touch/tuse.py --tag cx2e --thr 0.3 --w 1 --hold 3 --keep 0

Run outputs are copied to `/mnt/project-files/alldata/context/runs/`.
