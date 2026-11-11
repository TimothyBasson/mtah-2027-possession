"""A learned possession labeller: boosted trees score each frame, a sticky chain decodes the half.

The design follows the ball-free event-detection line of work (Kim et al., "Ball trajectory
inference from multi-agent sports contexts", KDD 2023; Kim et al., "PathCRF", arXiv 2602.12080):
learn per-frame scores for the possession state from the players' trajectories, then decode the
whole sequence with a chain model and Viterbi so that the output is a set of possessions and not
a flicker of frames. Here the states are the three the challenge grades (H, A, D) instead of a
player-to-player edge, the per-frame scorer is gradient-boosted trees on hand-built
team-level features (seq/features.py) instead of a set transformer, and the ball is used where it
is seen instead of never.

Unlike hmm/, nothing here is a hand-set or counted table: the per-frame scorer is fitted on
labelled matches from other games (external/, which RULES.md allows), and the chain has three
numbers (WEIGHT, POSSESSION_SECONDS, STOPPAGE_SECONDS).

The ball-alone features (k_: its speed, whether it is seen, how near a line) are built but left
out by default (external/score_seq.py --drop). IDSSE freezes the ball in a stoppage and Metrica
hides it, and a scorer fitted on those two learns exactly that: with k_ in, held-out SkillCorner
drops from 0.414 to 0.359 spellF1 and its frame accuracy from 0.80 to 0.71. Without k_, IDSSE is
unchanged (0.474 both ways), so nothing real is lost there.

On its own chain the labeller is ahead of hmm_v6 on two sets and behind on Metrica. Its use is as one more clue in the
HMM (external/score_seq.py, "hybrid"), where the two make different mistakes.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm.model import STATES, rate_log_trans, viterbi  # noqa: E402
from seq import features  # noqa: E402

STRIDE = 5                    # frames between scored frames (5 Hz at 25 fps)
THREADS = 4                   # fixed, so the fit does not depend on the machine's core count
WEIGHT = 0.5                  # trust in each scored frame's log-probability
POSSESSION_SECONDS = 12.0
STOPPAGE_SECONDS = 8.0
TO_DEAD_SHARE = 0.4
START = np.array([0.1, 0.1, 0.8])
CODES = {"H": 0, "A": 1, "D": 2}


def featurise(d, ball, swap=False, stride=STRIDE):
    """(X, names, frames) for one match; swap=True names the teams the other way round.
    d["team"] is one team per slot, or a (frames, slots) map (hmm/clues.team_by_frame)."""
    team = 1 - np.asarray(d["team"]) if swap else d["team"]
    return features.compute(d["players"], ball, team, [tuple(h) for h in d["halves"]],
                            d["pitch"], int(d["fps"]), stride=stride)


def fit(X, y, sample_weight=None, max_iter=300, seed=0):
    """The per-frame scorer. No subsampling and no early stopping, a fixed seed and a fixed thread
    count (THREADS), with scikit-learn pinned in requirements-seq.txt, so a refit on the same rows
    gives the same trees."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=False, random_state=seed)
    with threadpool_limits(limits=THREADS):
        return clf.fit(X, y, sample_weight=sample_weight)


def probabilities(clf, X, X_swapped):
    """p(H, A, D) per row, averaged with the swapped-team view so neither name is favoured."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREADS):
        p, q = clf.predict_proba(X), clf.predict_proba(X_swapped)
    return 0.5 * (p + q[:, [1, 0, 2]])


def decode(prob, frames, n_frames, halves, fps, stride=STRIDE, weight=WEIGHT,
           possession=POSSESSION_SECONDS, stoppage=STOPPAGE_SECONDS, to_dead=TO_DEAD_SHARE,
           min_possession=0.0, min_stoppage=0.0):
    """Labels for every frame: one Viterbi path per half over the scored frames, each scored
    frame's label held for the stride around it. X outside the halves."""
    rate = fps / stride
    trans = rate_log_trans(possession * rate, stoppage * rate, to_dead)
    log_emit = weight * np.log(np.clip(prob, 1e-6, 1.0))
    labels = np.full(n_frames, "X", dtype="<U1")
    for lo, hi in halves:
        sel = np.flatnonzero((frames >= lo) & (frames <= hi))
        if len(sel) == 0:
            continue
        path = viterbi(log_emit[sel], trans, np.log(START))
        if min_possession or min_stoppage:
            path = absorb_short(path, int(round(min_possession * rate)),
                                int(round(min_stoppage * rate)))
        idx = np.clip((np.arange(lo, hi + 1) - frames[sel[0]] + stride // 2) // stride, 0,
                      len(sel) - 1)
        labels[lo:hi + 1] = STATES[path[idx]]
    return labels


def absorb_short(path, min_possession, min_stoppage):
    """Give each run shorter than its minimum to the longer of its neighbours, shortest first."""
    path = path.copy()
    while True:
        edges = np.flatnonzero(np.diff(path)) + 1
        starts, stops = np.r_[0, edges], np.r_[edges, len(path)]
        if len(starts) < 2:
            return path
        lengths = stops - starts
        limit = np.where(path[starts] == 2, min_stoppage, min_possession)
        short = np.flatnonzero(lengths < limit)
        if len(short) == 0:
            return path
        i = short[np.argmin(lengths[short])]
        left = lengths[i - 1] if i > 0 else -1
        right = lengths[i + 1] if i + 1 < len(starts) else -1
        path[starts[i]:stops[i]] = path[starts[i - 1]] if left >= right else path[starts[i + 1]]
