# alldata/oss: ideas from public possession repos, tested on our harness

Thread "Learn from open-source possession models" (2026-11-17). The question: is there anything in
the wider open-source ecosystem for ball possession from tracking data that we have not tried,
and does it help the all-data model held out by provider? Direct challenge entries are another
thread's job. **Result: nothing to take. The two untried ideas both score level (+0.000 to
+0.002), well under the +0.01 bar. The submission is unchanged.**

## What the public repos do

| Repo / paper | What it does for possession | Already covered by us? |
|---|---|---|
| [kloppy](https://github.com/PySport/kloppy) | Loads providers' tracking and events, including their own `ball_owning_team` flag. No inference. | Our loaders read the same flags. |
| [floodlight](https://github.com/floodlight-sports/floodlight) | Loads IDSSE / DFL possession and ball-status codes as `Code` objects; space, kinematics, metabolic models. No possession inference. | Same as above; IDSSE is one of our providers. |
| [databallpy](https://github.com/alek050/databallpy), [docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html) | Team possession from events: switches on a successful on-ball event (pass, dribble, shot) of the other team. Player possession after [Vidal-Codina et al. 2022](https://arxiv.org/abs/2202.00804): nearest player within 1.5 m, gain valid only if the ball changes speed by 5 m/s or direction by 10 degrees, same player's segments carried over. | Yes. The rule is the hand-set touch detector of `alldata/touch/` (adds nothing on the smoothed ball; the learned detector is what works) and close to the control-rule folding of `external/control.py`. Event-driven switches: BAR-ATH's Smart Tagging was checked in `tagcheck/` (tags wrong 23% in the public window). |
| [BallRadar](https://github.com/pientist/ballradar) ([Kim et al. 2023](https://arxiv.org/abs/2306.08206)) | Players only: set transformer + hierarchical Bi-LSTM infers the ball's path and the ball holder (78% holder accuracy within one dataset). | Yes: the players-only `deep/` nets, the pointer holder net (`alldata/holder/`, nothing added to the team call). |
| [TranSPORTmer](https://arxiv.org/abs/2410.17785) (Capellera et al., ACCV 2024) | Transformer over players and ball with a per-frame CLS token classifying pass / possession / uncontrolled / out of play (about 81% with the ball hidden). No public code found. | Partly: our two heads cover possession and dead; "pass" and "uncontrolled" as their own classes are what the touch detector and the change head approximate. The holder-net result (auxiliary per-player targets add nothing) makes a ball-state auxiliary head unlikely to move the team call; not trained. |
| [Peral et al. 2025](https://arxiv.org/abs/2512.19528) | TranSPORTmer plus video crops of each player, masked pre-training; 90% holder accuracy on a proprietary league. No code. | Needs player crops from video; BAR-ATH's broadcast feed was sized in "Propose next model improvements" (5 points more ball coverage). Not tried. |
| [PathCRF](https://arxiv.org/abs/2602.12080) ([code](https://github.com/hyunsungkim-ds/pathcrf), Kim et al. 2026) | Players only, trained on the public IDSSE data (Bassek et al. 2025). Predicts a possession path over player-to-player edges with a CRF: transitions are learned from the network per frame, and possession may only pass through a sender-receiver edge (no "teleporting"). Viterbi decode. | **Not tried before. Tested below (two ways).** |
| [tryolabs soccer-video-analytics](https://github.com/tryolabs/soccer-video-analytics) | Video: nearest player to the detected ball, team switches after a run of frames (inertia). | Yes: our switch cost and minimum possession length. |

Across these, the only mechanism we had not measured is PathCRF's: switch costs that the network
learns per frame, and possession changing hands only at a reception.

## Test 1: switch only at a touch (`gate.py`)

The shipped decode (all-data net `pw1e`, touch evidence weight 1 / hold 3 s) with a direct
H<->A switch made dearer (`hi`) everywhere except within `win` frames of a detected touch by the
incoming team (the touch detector of `alldata/touch/`, score x `b` off the cost). Saved held-out
probabilities, so it runs in 10 s. Picked on the validation matches, scored on test.

| test, as built / folded | IDSSE | Metrica | SkillCorner | mean |
|---|---|---|---|---|
| shipped decode (touch evidence) | 0.665 / 0.760 | 0.793 / 0.831 | 0.419 / 0.521 | 0.626 / 0.704 |
| hi 8, no gate (control: just stickier) | 0.663 / 0.761 | 0.794 / 0.831 | 0.420 / 0.521 | 0.626 / 0.704 |
| hi 8, gate b 2, win 3 | 0.663 / 0.760 | 0.794 / 0.833 | 0.420 / 0.519 | 0.625 / 0.704 |
| hi 12, gate b 2, win 3 (best on validation, +0.007 / +0.003 there) | 0.659 / 0.761 | 0.792 / 0.832 | 0.421 / 0.519 | 0.624 / 0.704 |

Level. On validation the small gain came from the higher cost alone (the no-gate control scores
the same), and it does not carry to the held-out provider.

## Test 2: learned switch costs from the net's change head (`fit_edge.py`, `edge_dec.py`)

Every all-data net already trains a change head (possession change within 2 rows, weight 0.3)
that nothing reads. `fit_edge.py` is `fit.py` unchanged except that it also saves that head's
probability. One seed of the shipped recipe (`--drop-cols tim --share skillcorner=0.25 --pseudo
w1e`, tag `ew1`, about 20 min a fold), then the decode lowers the H<->A cost by `b` x change
probability. Every row is the same net, so seed noise cancels.

| test, as built / folded | IDSSE | Metrica | SkillCorner | mean | vs its control |
|---|---|---|---|---|---|
| `ew1`, shipped decode without touches (control) | 0.648 / 0.741 | 0.782 / 0.817 | 0.411 / 0.516 | 0.614 / 0.691 | |
| + change head, hi 4, b 4 | 0.650 / 0.742 | 0.782 / 0.817 | 0.411 / 0.516 | 0.614 / 0.691 | +0.001 / +0.000 |
| + change head, hi 8, b 8 | 0.650 / 0.744 | 0.779 / 0.817 | 0.413 / 0.516 | 0.614 / 0.692 | +0.002 / +0.001 |
| `ew1` with touch evidence (control) | 0.661 / 0.756 | 0.787 / 0.828 | 0.413 / 0.516 | 0.620 / 0.700 | |
| + change head, hi 8, b 8 | 0.661 / 0.755 | 0.784 / 0.824 | 0.414 / 0.516 | 0.620 / 0.699 | +0.001 / -0.000 |

Level. The change head says what the class probabilities already say: its peaks sit where the
team probabilities cross, so lowering the cost there moves no switch the decode was not already
making. A second seed was started and stopped once the first showed nothing. `ew1` alone scores
0.614 / 0.691 against the two-seed `pw1e` 0.616 / 0.696, so the copy of the recipe reproduces.

## Reading

The public tools are either loaders (kloppy, floodlight), rule sets we already have in a better
learned form (databallpy / Vidal-Codina, tryolabs), or players-only networks trained and scored
within one data source (BallRadar, TranSPORTmer, PathCRF) whose reported holder accuracies (70 to
90%) are not held out by provider. None attacks the loss that is left (brief possessions given to
the other team, where providers disagree about which touches count), and the one structural idea
we lacked, learned and touch-gated transitions, measures level.

## Commands

    uv sync --group deep && uv pip install numba      # numba for the fast Viterbi in turnover/common.py
    # data: /mnt/project-files/data/external -> data/external; alldata/learned/runs/{pw1e,w1e} -> data/cache/alldata/runs;
    #       alldata/touch/runs/t1 -> data/cache/alldata/touch/runs; deep/bt -> data/cache/deep/bt
    python alldata/oss/gate.py --kind val; python alldata/oss/gate.py --kind test --hi 8 12 --b 0 2 4 --win 3
    python alldata/learned/build.py --no-tim
    python alldata/oss/fit_edge.py --fold idsse --tag ew1 --drop-cols tim --share skillcorner=0.25 --pseudo w1e   # and metrica, skillcorner
    python alldata/oss/edge_dec.py --tag ew1 --kind val; python alldata/oss/edge_dec.py --tag ew1 --kind test --hi 4 8 --b 0 4 8

Nothing outside `alldata/oss/` changes; no BAR-ATH file was made.
