# Keeper-end detector (negative result, 2026-11-17)

A small classifier of p(team defending a goal has the ball, other team, dead) from ball-light
features near each goal (features.py), trained with each provider held out (train.py) and added
to the v9 decoder's emissions as a log ratio on gated frames (clue.py, score.py).

Held out, external/score_fuse.py's yardstick, hold on; v9 rerun 0.729 / 0.807 / 0.496 = 0.6776:

| keeper weight | gate | mean |
|---|---|---|
| 0.25 / 0.5 / 1 | ball near a goal | 0.6718 / 0.6641 / 0.6492 |
| 0.5 / 1 / 2 | same, ball unseen only | 0.6762 / 0.6698 / 0.6578 |

No weight helps any provider, so the weight picked on the other two providers is 0. Not wired
into hmm/decode.py.

    python external/control_features.py && python external/train_control.py --loso
    python keeper/build.py && python keeper/train.py && python keeper/score.py
