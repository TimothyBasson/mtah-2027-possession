# Short possessions in the decode: can the last step stop merges?

Failure mode 2 of the final model's review: a missed brief touch merges the possessions either
side of it, so one miss costs up to three (BAR-ATH public window: touches of 0.8 s and 1.3 s cost
five possessions). This folder tests whether the step that turns frame probabilities into
possessions can put such touches back. **Nothing shipped changes**: `alldata/learned/` and its
files are untouched, and the result below does not support changing the submission's decode.

## Result

Held out by provider, the all-data model's saved probabilities (`pw1e`, BAR-ATH-like balls, two
seeds averaged, both heads averaged as `run.py` decodes them). No training is involved, so there
is no seed spread beyond the nets'. Columns: IDSSE / Metrica / SkillCorner / mean of providers.

| decode | labels as built | control-rule (folded) labels | triples merged, as built |
|---|---|---|---|
| shipped (cost 4, 1 s / 1 s) | 0.632 / 0.769 / 0.426 / **0.609** | 0.759 / 0.836 / 0.482 / **0.692** | 47% |
| cost, team-change cost, minimum lengths, picked on validation | 0.630 / 0.767 / 0.431 / 0.610 | 0.757 / 0.834 / 0.491 / 0.694 | 47% |
| explicit-duration (semi-Markov), picked on validation for folded | 0.631 / 0.745 / 0.437 / 0.604 | 0.757 / 0.833 / 0.501 / 0.697 | 47 to 50% |
| brief touches written in, peak 0.5 on the as-built head | 0.661 / 0.780 / 0.409 / 0.617 | 0.748 / 0.806 / 0.453 / 0.669 | 41% |
| same, peak 0.2 | 0.656 / 0.763 / 0.393 / 0.604 | 0.709 / 0.756 / 0.427 / 0.631 | 35% |
| same, threshold picked on validation for as built | 0.646 / 0.783 / 0.409 / 0.612 | 0.754 / 0.820 / 0.453 / 0.676 | |
| same, picked on validation for folded | no touches written: the shipped decode | | |
| learned scorer of candidate touches, taught on as built | 0.663 / 0.766 / 0.405 / 0.611 | 0.721 / 0.775 / 0.449 / 0.648 | 38% |
| learned scorer, taught on folded | 0.632 / 0.769 / 0.425 / 0.609 | 0.758 / 0.836 / 0.481 / 0.692 | 47% |

"Triples merged": of the brief (under 3 s) true possessions with the other team on both sides,
the share where one drawn possession runs from the middle of one neighbour to the middle of the
other.

Reading:

- **Tuning the plain decode is flat**, as before: no cost or minimum length moves the merge slice.
- **A duration prior does not help.** It makes the decode prefer plausible lengths, which removes
  short possessions rather than adding them; its small folded gain is SkillCorner only and it
  merges more, not less.
- **Writing touches in does cut merges, and is worth +0.029 on IDSSE and +0.011 on Metrica on the
  labels as built** (IDSSE up in 7 of 7 matches, Metrica 2 of 3), with SkillCorner down 0.017
  (up in 2 of 20). **On control-rule labels every provider goes down** (-0.011 / -0.030 / -0.029),
  because that labelling has folded most of those touches into their neighbours: a bump of 0.5 to
  0.7 on the as-built head is a true touch 77% of the time on IDSSE's labels as built and 59% on
  its folded labels (Metrica 45% and 18%, SkillCorner 27% and 20%).
- **A learned scorer does not raise the precision**: fitted on the bump's shape in both heads, it
  matches the single threshold as built and, taught on folded labels, chooses to write nothing.
- So the limit is the evidence, not the decode. Of the brief sandwiched touches the net shows the
  right team at 0.5 or more inside 41 to 78% (IDSSE, Metrica; fewest under 1 s) and under 0.15
  inside 11 to 45%;
  the touches it does show are mostly already drawn, with the loss in their neighbours being
  given to the same team (the wrong-team mode, not this one).

By the bar set for this work (control-rule labels, no provider down) the insertion is **not
adopted**.

## BAR-ATH public window (a check, 17 possessions)

The two touches named in the review peak at 0.26 and 0.17 for the right team (0.35 and 0.24 on the
as-built head). Writing touches in at a peak of 0.15 to 0.2 finds both and scores 0.674 to 0.684
(13 of 17) against the shipped 0.613 (10 of 17), but that threshold costs 0.06 held out on
control-rule labels and draws 169 / 212 possessions per half against the shipped 131 / 150. At
the threshold the held-out as-built labels support (0.5) the window scores 0.565: neither touch is
reached and one false touch is added. `apply.py` writes that optional file (19 touches added,
passes `evaluation/validate_submission.py`); with `--tau 1` it reproduces the submission byte for
byte.

## Files

- `common.py` held-out probabilities, a compiled spellF1 equal to the organisers' (`python
  alldata/shortdec/common.py` checks it), the merge slice.
- `dec.py` the shipped decode compiled, a second minimum for sandwiched runs, `insert_touches`.
- `sweep.py` cost / minimum-length grids, picked on each fold's validation matches.
- `insert.py` touches written in at a threshold, held out and (`--bar`) on the public window.
- `learned.py` the learned scorer of candidate touches.
- `semimarkov.py` the explicit-duration decode.
- `apply.py` the optional BAR-ATH file.

Needs `data/cache/alldata/runs/pw1e` (copy in `/mnt/project-files/alldata/learned/runs/`),
`numba` and `scikit-learn`. Each script runs in under a minute on 4 CPUs.

Not measured: the held-out probabilities of the nets that read Tim's v8 (the submission's) were
never saved, so the held-out tables are on the plain all-data model, whose decode is the same.
