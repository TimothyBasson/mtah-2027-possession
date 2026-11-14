# Work in progress: joint control + seq classifier (not for merging as is)

Scratch scripts behind the held-out numbers quoted in the project thread. They will be replaced by
proper code before the pull request. Run from the repo root, after `external/control_features.py`:

```bash
P="uv run --frozen --with scikit-learn==1.6.1 python"
$P external/wip/prep.py     # hmm_v6 log-emissions + folded labels per match -> data/cache/wip/emit
$P external/wip/feat.py     # control + seq features, every 5th frame, both orientations -> data/cache/wip/joint
BALL="c_seen,c_ball_speed,c_bspeed,c_ball_dv,c_since_seen,c_line_at_last_seen,k_"
$P external/wip/train.py j1.npz --drop $BALL,d_,@-20,@+20,b_touch_balance --bal --reg   # held-out posteriors -> data/cache/j1.npz
$P external/wip/ev.py j1.npz      # folded spellF1 per provider at weights 0.05..0.3
$P external/wip/e3.py j1.npz      # the same with short spells folded and other transitions
```

`j1` scored 0.638 / 0.814 / 0.480 (IDSSE / Metrica / SkillCorner) at weight 0.15. The `--drop` above
also leaves out the columns added to seq/features.py after that run (touch balance, ball direction,
20 s windows), which are still being tested.

To augment: feat.py reads `data/external/<name>.npz` and `data/cache/control_<name>.npz`; write
augmented copies of both under other names and add them to the training rows in train.py.
