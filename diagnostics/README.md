# Where spellF1 is lost

Analysis only: nothing here changes a model. It labels the 30 outside matches with a possession
model exactly as `external/heldout.py` scores it (weights refitted without the provider being
labelled), then takes the score apart.

```bash
uv run python diagnostics/predict.py --model possession --out data/cache/diag/main   # ~5 min
uv run python diagnostics/analyze.py main                                             # ~1 min
uv run python diagnostics/report.py main
uv run python diagnostics/whatif.py      # paired noise between saved versions, stoppages redrawn
```

`--model` takes any folder with `model.py` / `features.py` (a worktree of another branch works), and
trailing `param=value` pairs are passed to the model. Predictions go under `data/cache/diag/<tag>/`.

What `report.py` prints, every mean with the three providers counted equally:

- the score, precision and recall by IoU, spread between matches and the standard error of the mean
- **repairs**: the score with one kind of mistake set right from the truth and the rest left alone.
  Team mistakes are sorted by where the wrong run sits in the true possession (the whole of it, its
  start, its end, the middle); alive/dead mistakes the same way. Gains overlap and do not add up.
- the truth passed through the model's own short-run absorbing: the ceiling that clean-up allows
- recall by true possession length and by how it starts and ends; precision by predicted length
- how early or late the model's turnovers, stoppage starts and restarts are
- the score with the whole prediction slid in time
