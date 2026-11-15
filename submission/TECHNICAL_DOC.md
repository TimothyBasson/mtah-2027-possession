<!--
DRAFT. Rename to Gambtt.md (or export Gambtt.pdf) when it is uploaded: SUBMISSION.md asks
for both files to carry the team name. *.pdf is git-ignored in this repo, so export the PDF into
submission/out/.

The six sections below are the ones SUBMISSION.md lists. "A few pages is plenty."
Lines starting "TODO" are decisions still open; delete them as they close.
-->

# Gambtt: who has the ball, frame by frame

**More than a Hack 2027, Phase 1: Ball Possession** · repository: TODO public URL · commit: TODO

## 1. Approach

TODO: one paragraph on what the final system is, once the method is chosen
(`method` in submission/submission.json).

TODO: why that one rather than the alternatives. The ranking metric is spellF1 (whole possessions
matched by temporal IoU), so say how the method is built to get possession *count and boundaries*
right rather than per-frame accuracy.

Candidates still in play:

- **HMM** (`hmm/`): sticky hidden Markov model over hand-built tracking clues (nearest-player gap
  and distance, ball place, player speed, flight, team width) plus a learned per-frame control
  classifier, decoded with Viterbi per half.
- **Ball-taught network** (`deep/`, tag `bt`): players-only network whose training is supervised
  by the ball on outside matches; the ball is never read on BAR-ATH.

## 2. Assets used, and how

| asset | used? | how |
|---|---|---|
| tactical-camera automatic tracking (ATD) | TODO | |
| broadcast tracking | TODO | |
| Smart Tagging | TODO | |
| homography | TODO | |
| video | TODO | |

Tried and dropped: TODO (list them; SUBMISSION.md says this is as useful as what was kept).

## 3. Training data

None of it comes from BAR-ATH. TODO: confirm this table matches the final method.

| dataset | source / licence | labelled by | amount |
|---|---|---|---|
| IDSSE | [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0; Bassek et al., *Scientific Data* 12, 195 (2025) | data provider (official event + tracking data) | 7 matches |
| Metrica sample data | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) @ `e706dd5` | data provider | 3 matches |
| SkillCorner open data | [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT; A-League 2024-25 | data provider | 20 matches |

TODO: how possession labels were derived from those providers' data (external/README.md), and any
label folding (external/control.py).

### What touches BAR-ATH, and how

"None of it comes from BAR-ATH" covers human labels. Some parts of the pipeline still *read* the
match, so they are listed here one by one. TODO: confirm each line for the final method, and
delete the lines for the method not chosen.

| component | learned from | reads BAR-ATH? | uses BAR-ATH labels (incl. public 5 min)? |
|---|---|---|---|
| `hmm/models/control_state.npz` (HMM's control classifier) | the 30 outside matches only (hmm/models/README.md) | applied to its tracking at decode time | no |
| HMM tables and transition settings | counted from the outside matches' labels | no | no |
| `--fit-place` (HMM, **off by default**) | the match's own tracking, from what the other clues say; unsupervised, no labels | yes | no |
| `deep/models/bt.pt` | the 30 outside matches only; decode settings picked on held-out outside folds (deep/README.md) | applied to its players' tracking; its ball is never read | no |
| public 5-minute ground truth | — | — | scored as a pipeline check only; TODO confirm no weight, threshold or method choice was picked on it |

## 4. Possession rules implemented

TODO, for the final method:

- **Loose balls**: when does possession pass, at the touch or at the landing of a flight?
- **Challenges / duels**: what happens to very short spells between two spells of the other team?
- **Restarts and stoppages**: what makes a frame `D`, and what happens before each kickoff?
- **Abstentions**: does the file ever write `X` on a graded frame?

## 5. Limitations

TODO: where it fails (measured on outside matches, not on the public 5 minutes), and what we would
fix with more time.

## 6. Reproduction

```bash
# 1. data: unpack the organisers' bundle into data/ (see data/README.md); external matches: external/README.md
# 2. environment: TODO pin one (Python version + requirements files for the chosen method)
pip install -r requirements.txt            # TODO: plus requirements-deep.txt if method = deep
# 3. build, validate and fingerprint the file
python submission/make_submission.py --twice
```

Expected output: `submission/out/Gambtt.csv` with sha256 `TODO` (from the manifest written next to it).

TODO: the sha256 above must come from a **clean clone in the pinned environment**, not from the
development checkout. `--twice` only shows the method is deterministic on one machine; the rules
ask for the same bytes on the organisers' run.
