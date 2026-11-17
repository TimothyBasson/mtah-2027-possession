<!--
Mirror of the team's working copy (a Claude Doc, where it is commented on and exported to the
upload PDF), as of 2026-11-17. If the two differ, the Claude Doc is newer: paste its final text
over this file before the repo goes public. This comment is left out by build_doc_pdf.py.
-->

# Gambtt: who has the ball, frame by frame

**More than a Hack 2027, Phase 1: Ball Possession.** Repository: *open: public URL*. Commit: *open*.

## 1. Approach

Every frame is labelled by **hmm_v9**: a sticky hidden Markov model with three states (home in possession, away in possession, dead ball), decoded with Viterbi once per half. Held out by provider on 30 outside matches it scores a mean spellF1 of 0.677, against 0.647 for the version before it.

Each frame's evidence is a sum of log-likelihood terms:

- **Hand-built tracking clues**, with tables counted from the outside matches' labels: the gap and distance between the nearest home and away players and the ball; where the ball is (in play, past a line, lost at a line, lost in play); how fast the players move (when the ball is dead they walk); flights (after a kick, the kicker's team keeps the clue until the next touch); and which team is spread wider.
- **A control classifier** (gradient-boosted trees over 248 features: both teams' nearest players, how they move with the ball, touches, windows before and after the frame, and the shape of all 22 players), giving p(H, A, D) per frame at weight 0.2.
- **Two players-only neural nets** (`bt` and `sw25`, averaged, weight 0.3) that read all 22 players and never the ball. They speak on the 41% of graded frames where the ball is missing, which is where the version without them lost most.
- **A keeper-area clue** (weight 4x) for a ball lost in play in a keeper's own box that turns up in his hands.

Two rules then adjust stoppage edges without adding or removing possessions. The **restart hold** moves the start of a spell after a stoppage to the kick of a ball that was lying still. The **keeper hold** gives a short stoppage back to the keeper's team when he gathered the ball and his team plays on.

**Why this design.** spellF1 scores whole possessions, so the model has an explicit cost for changing state. Per-frame classifiers and short smoothing windows fragment possession: the organisers' raw baseline reports 2,265 possessions. The HMM lets clues of different strength be weighed against each other per frame, and each component was kept only if it raised spellF1 held out by provider. The players-only net alone in the same decoder scores 0.660; together with the control classifier, 0.677.

### Results

Folded spellF1 (see section 3 for the folding), each provider scored by models that never saw it:

| Version | Change | IDSSE (7) | Metrica (3) | SkillCorner (20) | Mean |
| --- | --- | --- | --- | --- | --- |
| v9 as submitted | + keeper clue, restart and keeper holds | 0.727 | 0.807 | 0.496 | 0.677 |
| v9 | + two players-only nets | 0.728 | 0.807 | 0.497 | 0.677 |
| v8 | control classifier retrained, ball-free features | 0.644 | 0.816 | 0.480 | 0.647 |
| v7 | + control classifier | 0.609 | 0.798 | 0.428 | 0.612 |

The net weight was read off these matches. Picked for each provider on the other two, the mean is **0.664**, which is the fair estimate. The keeper clue and the two holds are flat held out (0.6764 to 0.6767), because the outside providers hide or freeze the ball in exactly the situations they target.

On BAR-ATH, with no labels read, v9 finds 241 possessions (117 / 124 per half, mean 14.9 / 13.3 s), measured before the keeper rules; the public window's mean is 14.2 s. Of the 58 set pieces in Smart Tagging, 53 have a decoded restart within 2 s.

On the public 5 minutes, scored once as a pipeline check: spellF1 **0.774** (bootstrap 5-95%: 0.58 to 0.90), macroF1 0.904, 14 possessions against 17 true. The interval is wide; it shows nothing has gone wrong, not that one version beats another.

## 2. Assets used, and how

The labels come from the tactical-camera tracking alone; the other assets are used for checks or not at all.

| Asset | Used? | How |
| --- | --- | --- |
| Tactical-camera automatic tracking (ATD) | yes, the only input | players and ball positions for every clue, the control classifier and the nets (the nets read the players only) |
| Smart Tagging | kickoff frames; checks | the published kickoff frames pin graded frames before each kickoff to dead ball; the tagged set pieces are printed beside decoded restarts as a diagnostic, never used to set a label |
| Broadcast tracking | no | |
| Homography | no | the ATD already gives pitch positions (normalised 0-1, scaled to a 105.3 x 68 m pitch); homography only maps them to video pixels |
| Video | for error analysis only | watched to find failure types (restarts called live early; keepers holding a ball the tracker loses). The rules built from that are set on the outside matches; no frame, time or constant from BAR-ATH is in the code |

**Tried and dropped**, each measured held out on the outside matches:

- Counting the ball-place table on the match itself (`--fit-place`): +0.007, interval spans zero.
- The control classifier replacing the gap and flight clues instead of joining them: worse on every provider.
- Cheaper team switches where a second classifier sees a change of possession: helps raw labels, not folded ones; costs SkillCorner.
- The nets' outputs as 45 extra classifier features: 0.658, well under using the net in the decoder.
- A look-ahead clue reading every missing-ball gap by where the ball turns up next: removed some false stoppages but cost held-out score; only its keeper part was kept.
- Folding very short possessions into the surrounding team's spell, and turning short spells between two stoppages into dead ball: flat or worse.
- A whole-match context second stage, and self-training on the unlabelled target match: no gain.

## 3. Training data

Every learned part of v9 (the clue tables, the control classifier and both nets) is trained on the same 30 outside matches. No human label on BAR-ATH is used anywhere.

| Dataset | Source and licence | Labelled by | Amount |
| --- | --- | --- | --- |
| SkillCorner open data | [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT; A-League 2024-25 | the provider: phases of play and per-frame possession group | 20 matches |
| IDSSE | [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0; Bassek et al., Scientific Data 12, 195 (2025) | DFL data operators: per-frame BallStatus and BallPossession | 7 matches (2 Bundesliga, 5 2. Bundesliga, 2022-23) |
| Metrica sample data | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) at commit `e706dd5` | rebuilt by our code from the provider's event data | 3 matches |

**How the labels are used.** None of the three sources applies the challenge's controlled-possession rule; DFL's labels flip on a first touch, so 20-30% of their possessions are under 2 s. We therefore also score against **folded** labels (external/control.py): a spell under 2 s, between spells of the other team or a stoppage, with fewer than 2 touches by its own team, goes back to the team that had the ball. The 2 s and 2 touches come from the control test in the organisers' DATA_POINTS.md, not from any ground truth. The control classifier is trained on folded labels.

To make the outside matches look like BAR-ATH, the IDSSE and SkillCorner balls are knocked out to 59% seen with a fixed-seed dropout before features are computed. SoccerTrack v2 (8 matches, CC BY 4.0) was tried as a fourth training source for the nets and not kept.

### What reads BAR-ATH, and how

Some parts of the pipeline read the match being labelled. None of them uses a BAR-ATH label.

| Component | Learned from | Reads BAR-ATH? | Uses BAR-ATH labels (incl. public 5 min)? |
| --- | --- | --- | --- |
| Control classifier (`hmm/models/control_state.npz`) | the 30 outside matches | applied to its tracking | no |
| Clue tables and transition settings | counted from the outside matches' labels | no | no |
| Players-only nets (`deep/models/bt.pt`, `sw25.pt`) | the 30 outside matches, 2,000 steps | applied to its players' tracking; its ball is never read | no |
| Restart hold and keeper hold | parameters swept on the outside matches | applied after decoding | no |
| Public 5-minute ground truth | not used for learning | not applicable | scored as a pipeline check only |

No setting was tuned on the public window; the code records this beside each one (`hmm/decode.py`, `hmm/ball.py`, `deep/README.md`, `external/README.md`). The choice of hmm_v9 over the nets alone was made on the held-out outside matches (0.677 against 0.660 in the same decoder).

## 4. Possession rules implemented

Possession changes only when the other team gains control, following the organisers' rule; no rule is hard-coded per frame, the model weighs evidence instead.

- **Loose balls:** possession passes on control, not on the first touch. During a flight the kicker's team keeps the evidence until the next touch, so a long ball changes possession where it lands, not where it was struck. A ball with nobody near it is weak evidence for either team and leans towards a stoppage.
- **Challenges and duels:** the HMM's cost for changing state, and a control classifier trained on folded labels (under 2 s and under 2 touches is not a change), keep possession with the team in control through a deflection or a contested touch. Very short possessions are not removed by a rule: that was measured and was flat.
- **Restarts and stoppages:** a frame is dead ball when the ball is seen past a line or lost at one, when the players walk, or when nobody is near it. Graded frames before each half's kickoff are dead. After a stoppage, possession starts at the kick of a placed ball (restart hold). A keeper holding a ball the tracker has lost is his team's possession, not a stoppage (keeper clue and keeper hold).
- **Abstentions:** the file writes `X` only outside the two halves' playing windows (before kickoff, half-time, after the final whistle). Every frame inside them gets `H`, `A` or `D`.

## 5. Limitations

The weakest part is naming the team in short possessions where the ball is missing. All numbers here are from the outside matches unless marked BAR-ATH.

- **Short possessions of the other team.** Held out, most spell errors are 2-5 s possessions of the other team, missed and invented in similar numbers (IDSSE 116 missed / 30 invented, SkillCorner 293 / 428). That is team-naming quality, not a timing bias a rule can move.
- **Broadcast-style tracking.** SkillCorner, tracked from broadcast video, scores 0.496 against 0.727 and 0.807 for the other two.
- **Dead-ball share on BAR-ATH.** v9 calls 45-48% of graded frames dead per half, against about 20% in the public 5 minutes. The first minutes may not be typical, but some long ball-missing gaps are probably called dead that were live. The keeper rules move the share from 0.470 to 0.462.
- **Restarts the hold cannot reach:** throw-ins (the ball moves with the thrower) and restarts with the ball unseen.
- **The held-out yardstick is blind in places.** The outside providers hide or freeze the ball in stoppages and never lose it in a keeper's hands, so the restart and keeper rules score flat there; they were checked on BAR-ATH's video, which is evidence of a kind we could not score.
- **Reproducibility across CPUs.** Two runs on one machine give the same bytes; the file has not been regenerated on a second CPU, and the nets' last bits could differ between instruction sets.

**With more time:** better team naming in short ball-missing possessions; calibrating the dead-ball call to this tracker's ball dropouts; a regeneration check on a second machine.

## 6. Reproduction

One command builds, validates and fingerprints the file. It runs on a CPU; no GPU is needed.

1. **Data:** unpack the organisers' bundle into `data/` (data/README.md). Only the tactical tracking, its metadata and Smart Tagging are read.
2. **Environment:** Python 3.11, then `pip install -r requirements.txt -r requirements-deep.txt` (numpy, pandas and torch 2.15.0, pinned; `uv.lock` pins every version).
3. **Build:** `python submission/make_submission.py --twice`. It runs `python -m hmm.decode`, runs the organisers' validator, checks that two runs give identical bytes and writes `submission/out/Gambtt.csv` with a manifest of its sha256, commit and library versions.

The committed models are read, not retrained: `hmm/models/control_state.npz`, `deep/models/bt.pt` and `deep/models/sw25.pt`. Retraining is documented (external/README.md) but a refit is not byte-identical, so the file is reproduced from the committed models.

As of the last recorded build on main, `hmm_v9.csv` has sha256 `26f1cbce...e951a35f4`. *Open: replace this with the full sha256 and commit from the final build's manifest.*
