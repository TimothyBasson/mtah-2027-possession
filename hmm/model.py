"""A three-state sticky HMM over H / A / D, in log space, numpy only.

Everything here is generic: it knows nothing about football beyond the three state indices. The
football lives in the numbers handed to it -- see hmm/decode.py.

    log_emit[t, s]   log-likelihood of frame t's clues if the true state is s
    log_trans[i, j]  log p(state j at t+1 | state i at t)
    log_start[s]     log p(state s on the first frame)

Viterbi returns the single most probable path for a whole half. It reads future frames, which is
allowed here: the submission is produced offline.
"""

import numpy as np

H, A, D = 0, 1, 2
STATES = np.array(["H", "A", "D"])


def sticky_log_trans(p_switch, n_states=3):
    """Stay with probability 1 - p, leave to each other state equally. p = e^-lambda."""
    probs = np.full((n_states, n_states), p_switch / (n_states - 1))
    np.fill_diagonal(probs, 1.0 - p_switch)
    return np.log(probs)


def rate_log_trans(possession_frames, stoppage_frames, to_dead_share):
    """Transitions from the mean length of a possession and of a stoppage, in frames.

    A run that leaves with probability q per frame lasts 1 / q frames on average, so the two
    durations fix the diagonals. Of the possessions that end, `to_dead_share` end in a stoppage
    and the rest go straight to the other team. A stoppage hands the restart to either side
    equally -- v1 has no clue for who restarts.
    """
    leave = 1.0 / possession_frames
    restart = 1.0 / stoppage_frames
    probs = np.array([
        [1.0 - leave, leave * (1.0 - to_dead_share), leave * to_dead_share],
        [leave * (1.0 - to_dead_share), 1.0 - leave, leave * to_dead_share],
        [restart / 2.0, restart / 2.0, 1.0 - restart],
    ])
    return np.log(probs)


def build_log_emit(clues, n_states=3):
    """log_emit[t, s] = sum_i w_i * log_table_i[s, obs_i[t]], skipping obs < 0.

    Each clue is (obs, log_table, weight): obs is an int category per frame with -1 for "not
    observed", log_table is states x categories. The weight is a trust exponent -- consecutive
    frames at 25 fps are far from independent, and a weight below 1 stops the decoder from
    counting the same look at the ball twenty-five times a second.
    """
    out = None
    for obs, log_table, weight in clues:
        obs = np.asarray(obs)
        if out is None:
            out = np.zeros((len(obs), n_states))
        seen = obs >= 0
        out[seen] += weight * log_table[:, obs[seen]].T
    return out


def viterbi(log_emit, log_trans, log_start):
    """Most probable state path. O(T * S^2) with the time loop in Python, S^2 vectorised."""
    n_frames, n_states = log_emit.shape
    back = np.empty((n_frames, n_states), dtype=np.int8)
    score = log_start + log_emit[0]
    for t in range(1, n_frames):
        candidates = score[:, None] + log_trans  # from i (rows) to j (cols)
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(n_states)] + log_emit[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _logsumexp(x, axis):
    peak = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(x - peak), axis=axis))


def forward_backward(log_emit, log_trans, log_start):
    """Posteriors p(state_t | all frames), summed transition posteriors, and the log-likelihood.

    Scaled forward-backward: each step is normalised so nothing underflows over 80k frames.
    """
    n_frames, n_states = log_emit.shape
    trans = np.exp(log_trans)
    emit = np.exp(log_emit - log_emit.max(axis=1, keepdims=True))
    emit_shift = log_emit.max(axis=1)

    alpha = np.empty((n_frames, n_states))
    scale = np.empty(n_frames)
    a = np.exp(log_start) * emit[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, n_frames):
        a = (alpha[t - 1] @ trans) * emit[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]

    beta = np.empty((n_frames, n_states))
    beta[-1] = 1.0
    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_frames - 2, -1, -1):
        b = emit[t + 1] * beta[t + 1]
        xi_sum += alpha[t][:, None] * trans * b[None, :] / scale[t + 1]
        beta[t] = (trans @ b) / scale[t + 1]

    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True)
    loglik = float(np.sum(np.log(scale)) + np.sum(emit_shift))
    return post, xi_sum, loglik


def baum_welch(obs, log_table, log_trans, log_start, n_iter=20, weight=1.0, segments=None):
    """Unsupervised EM for one categorical clue: re-estimates the emission table and transitions.

    `segments` is a list of (start, stop) slices decoded independently (the two halves). The start
    prior is held fixed -- with two halves there is nothing to learn it from. Returns the fitted
    log_table, log_trans and the log-likelihood after each iteration.
    """
    obs = np.asarray(obs)
    segments = segments or [(0, len(obs))]
    n_states, n_cats = log_table.shape
    history = []
    for _ in range(n_iter):
        counts = np.zeros((n_states, n_cats))
        xi_total = np.zeros((n_states, n_states))
        loglik = 0.0
        for lo, hi in segments:
            seg = obs[lo:hi]
            log_emit = build_log_emit([(seg, log_table, weight)], n_states)
            post, xi_sum, ll = forward_backward(log_emit, log_trans, log_start)
            loglik += ll
            xi_total += xi_sum
            seen = seg >= 0
            np.add.at(counts.T, seg[seen], post[seen])
        history.append(loglik)
        log_table = np.log(counts / counts.sum(axis=1, keepdims=True))
        log_trans = np.log(xi_total / xi_total.sum(axis=1, keepdims=True))
    return log_table, log_trans, history
