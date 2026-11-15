# Committed model

`control_state.npz` is the v8 control classifier (hmm/control.py), exported to plain arrays by
hmm/trees.py. It holds only tree arrays, the 248 feature names and the class prior. It loads with
`allow_pickle=False` and nothing in it comes from BAR-ATH.

| | |
|---|---|
| sha256 | `577f70e1aaf6b9b20f2b7f8223fddc8f776b6fdb2e84c9fb3e86bb7d01471afd` |
| Trained by | `python external/control_features.py && python external/train_control.py` |
| Library | scikit-learn 1.6.1 (requirements-dev.txt), HistGradientBoostingClassifier, 300 rounds, no early stopping, `random_state=0` |
| Target | per-frame H / A / D, folded towards the control rule (external/control.py, 2 s, 2 touches); for SkillCorner the phase-of-play team (`train_labels`, external/build.py) |
| Training data | the 30 external matches in data/external (external/README.md), every 5th frame where labelled, both team orientations, the three providers weighted equally |

The v7 model (94 columns, sha256 `4223a1c4...`) is at commit fe457a3; the v8 features leave out
columns it needs, so this decoder cannot run it.

**A refit is not byte-identical.** v7's file was. A second fit of v8 from the same cached features,
same code and same pinned scikit-learn (4 threads both times) gave sha256 `318d4069...`: the same
900 trees in shape and count, equal up to tree 189, then leaf values that differ in the last
float digits, and from there a few different splits. On two matches' features the two models'
probabilities differ by 0.013 at most and 0.00001 on average, and name the same state on every
row. The likely cause is the order threads add up the weighted histograms in (v7 had no sample
weights). So the submission is reproducible from this committed file, which is what the decoder
reads, and not from a retrain.
`train_control.py` prints the sha256 so that you can check.

Training data sources (external/README.md has how to fetch them):

- IDSSE, 7 matches: [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data),
  CC BY 4.0. Bassek et al., *Scientific Data* 12, 195 (2025).
- Metrica, 3 sample games: [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
  at commit `e706dd506b360d69d9d123d5b8026e7294b13996`.
- SkillCorner, 20 matches: [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT.
  SkillCorner Open Data (A-League 2024-25).

The IDSSE and SkillCorner balls are thinned to 59% seen with the fixed-seed dropout in
external/score.py (`drop_ball`, seed 0) before the features are computed, as in
external/score_hmm.py.
