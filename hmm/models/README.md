# Committed model

`control_state.npz` is the v7 control classifier (hmm/control.py), exported to plain arrays by
hmm/trees.py. It holds only tree arrays, the 94 feature names and the class prior. It loads with
`allow_pickle=False` and nothing in it comes from BAR-ATH.

| | |
|---|---|
| sha256 | `4223a1c46e92ad2402ce4ff52289c2d36a6e82cb1c3e52ddd2d330ac600bc8f9` |
| Trained by | `python external/control_features.py && python external/train_control.py` |
| Library | scikit-learn 1.6.1 (requirements-dev.txt), HistGradientBoostingClassifier, `random_state=0` |
| Target | per-frame H / A / D, folded towards the control rule (external/control.py, 2 s, 2 touches) |
| Training data | the 30 external matches in data/external (external/README.md), every 5th labelled frame, both team orientations |

Retrained from the same features with the pinned scikit-learn, the file comes out byte-identical.
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
