"""The players-only net of deep/ as a clue in the decoder (hmm_v9).

The net (deep/net.py, the ball-taught "bt" kind) reads the players and nothing else, and gives
p(H, A, D) for every 5th frame. log_ratio() turns that into a scaled likelihood, as
hmm/control.py does for the control classifier, and the decoder adds it to its log emissions at
NET_WEIGHT (hmm/decode.py). The two models err in different places: the classifier follows the
ball and is lost where the ball is missing, the net reads all 22 players and is blind to short
possessions. Held out by provider the sum scores 0.674 against 0.647 for v8 (external/README.md).

Tried and not kept: the net's outputs as columns of the control classifier instead
(net_outputs(), half_columns(), external/net_features.py, train_control.py --net): its p(H, A, D),
its pointer's lean, and how the pointer and its guess of the ball sit against the visible ball.
That scores 0.658, and no better than this module's sum when both are used.

The net's own inputs never include the ball (tests/test_deep_no_ball.py).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = np.array([52.5, 34.0], np.float32)
FAR = 30.0   # metres: cap on distances, as hmm/control.py
NEAR = 3.0   # metres: "on the ball"
# 5 fps samples either side for the windowed means, and for the one-sided means.
WINDOWS = ((5, "1s"), (15, "3s"))
SIDED = (10, "2s")


# Class shares (H, A, D) of the labels the shipped net was trained on: the 30 outside matches as
# labelled, the three providers counting equally, H and A made equal (external/score_fuse.py
# label_shares). The net's posterior is divided by these before it joins the emissions.
NET_PRIOR = np.array([0.324, 0.324, 0.352])
STEP = 5  # the net's frame spacing (deep/data.py STEP)


def upsample(p, n, step=STEP):
    """(T, 3) on every step-th frame -> (n, 3) by linear interpolation (deep/decode.py)."""
    t = np.arange(len(p)) * step
    return np.stack([np.interp(np.arange(n), t, p[:, k]) for k in range(p.shape[1])], 1)


def log_ratio(probs, halves, n_frames, prior=NET_PRIOR, floor=1e-6):
    """(n_frames, 3) log p(state | players) - log p(state), 0 outside the halves. probs is one
    (T, 3) array per half, or a list of such lists (several nets, averaged); halves are
    (first, last) frames, last included."""
    if isinstance(probs[0], (list, tuple)):
        probs = [np.mean([np.asarray(p[i]) for p in probs], axis=0) for i in range(len(halves))]
    out = np.zeros((n_frames, 3))
    for p, (a, b) in zip(probs, halves):
        a, b = int(a), int(b)
        out[a:b + 1] = np.log(np.clip(upsample(np.asarray(p), b - a + 1), floor, 1)) - np.log(prior)
    return out


def _deep():
    path = os.path.join(ROOT, "deep")
    if path not in sys.path:
        sys.path.insert(0, path)
    import data
    return data


def net_outputs(net, players, ball, team, a, b, pitch, fps=25):
    """Run the net over frames a..b (b included) in its four symmetric views and average.

    Returns a dict on the net's 5 fps frames: p (T, 3) p(H, A, D); ptr (T, K) the pointer's share
    on each player slot the net was shown; xy (T, K, 2) those players and ball_hat (T, 2) the
    net's guess of the ball, in metres from the centre spot with home defending x < 0; mask (T, K);
    home (T, K) bool; edge, touch (T,) probabilities; free (T,) metres from the ball to the
    nearest player as the net guesses it; ball (T, 2) the ball as given, same frame as xy.
    """
    import torch
    import torch.nn.functional as F
    D = _deep()
    X, M, bxy, frames = D.half_tensors(players, ball, team, a, b, pitch, fps)
    out = dict(p=0, ptr=0, ball_hat=0, edge=0, touch=0, free=0)
    dummy = np.zeros(len(X), np.int64)
    with torch.no_grad():
        for s in (0, 1):
            for m in (0, 1):
                x = X
                if s:
                    x = D.swap_teams(x, dummy, bxy)[0]
                if m:
                    x = D.mirror(x, bxy)[0]
                logits, bhat, ehat, extra = net(torch.from_numpy(x)[None], torch.from_numpy(M)[None])
                p = F.softmax(logits[0], -1).numpy()
                out["p"] = out["p"] + (p[:, [1, 0, 2]] if s else p)
                bh = bhat[0].numpy().copy()
                if m:
                    bh[:, 1] *= -1
                if s:
                    bh = -bh
                out["ball_hat"] = out["ball_hat"] + bh
                out["edge"] = out["edge"] + torch.sigmoid(ehat[0]).numpy()
                out["ptr"] = out["ptr"] + (torch.softmax(extra["ptr"][0], -1).numpy()
                                           * M.any(-1, keepdims=True))
                out["touch"] = out["touch"] + torch.sigmoid(extra["touch"][0]).numpy()
                out["free"] = out["free"] + extra["free"][0].numpy()
    out = {k: v / 4 for k, v in out.items()}
    out["free"] = np.expm1(np.clip(out["free"], 0, 5))
    out["ball_hat"] = out["ball_hat"] * SCALE
    out.update(xy=X[..., :2] * SCALE, mask=M, home=X[..., 6] > 0, ball=bxy * SCALE, frames=frames)
    return out


def _mean(x, lo, hi):
    """NaN-aware mean of x over samples i + lo .. i + hi (both included)."""
    seen = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(seen, x, 0.0))])
    n = np.concatenate([[0], np.cumsum(seen)])
    i = np.arange(len(x))
    a = np.clip(i + lo, 0, len(x))
    b = np.clip(i + hi + 1, 0, len(x))
    b = np.maximum(a, b)
    cnt = n[b] - n[a]
    return np.where(cnt > 0, (c[b] - c[a]) / np.maximum(cnt, 1), np.nan)


def half_columns(o, swap=False):
    """((T, columns) float32, names) from net_outputs(), on the net's 5 fps frames. swap=True
    gives the same columns with the two teams' roles exchanged."""
    p = np.clip(o["p"], 1e-6, 1)
    ptr, mask = o["ptr"], o["mask"]
    home = o["home"] ^ bool(swap)
    lh, la, ld = np.log(p[:, 1 if swap else 0]), np.log(p[:, 0 if swap else 1]), np.log(p[:, 2])
    mass_h = (ptr * (mask & home)).sum(1)
    mass_a = (ptr * (mask & ~home)).sum(1)
    seen = np.isfinite(o["ball"][:, 0])
    d = np.linalg.norm(o["xy"] - np.nan_to_num(o["ball"])[:, None], axis=-1)
    d = np.where(mask, np.minimum(d, FAR), np.nan)
    rows = np.arange(len(d))

    def nearest(side):
        ds = np.where(mask & side, d, np.inf)
        k = ds.argmin(1)
        return k, ds[rows, k]

    kh, dh = nearest(home)
    ka, da = nearest(~home)
    top = ptr.argmax(1)
    with np.errstate(invalid="ignore"):
        cols = {
            "n_lh": lh, "n_la": la, "n_ld": ld, "n_team": lh - la,
            "n_dead": ld - np.maximum(lh, la),
            "n_mass_h": mass_h, "n_mass_a": mass_a, "n_mass": mass_h - mass_a,
            "n_ptr_top": ptr.max(1),
            "n_edge": o["edge"], "n_touch": o["touch"], "n_free": o["free"],
            # against the ball the decoder is shown (NaN where it is not seen)
            "n_ptr_dist": np.where(seen, np.nansum(ptr * np.nan_to_num(d), 1), np.nan),
            "n_ptr_near": np.where(seen, (ptr * (np.nan_to_num(d, nan=FAR) < NEAR)).sum(1), np.nan),
            "n_ptr_nearest_h": np.where(seen & np.isfinite(dh), ptr[rows, kh], np.nan),
            "n_ptr_nearest_a": np.where(seen & np.isfinite(da), ptr[rows, ka], np.nan),
            "n_top_dist": np.where(seen, np.nan_to_num(d[rows, top], nan=FAR), np.nan),
            "n_top_home": np.where(mask.any(1), np.where(home[rows, top], 1.0, -1.0), np.nan),
            "n_hat_dist": np.where(seen, np.minimum(
                np.linalg.norm(o["ball_hat"] - np.nan_to_num(o["ball"]), axis=-1), FAR), np.nan),
        }
        cols["n_ptr_nearest"] = cols["n_ptr_nearest_h"] - cols["n_ptr_nearest_a"]
        # the net's team lean signed by who is nearer the ball: positive when they agree
        near_home = np.where(seen & np.isfinite(dh) & np.isfinite(da), np.sign(da - dh), np.nan)
        cols["n_agree"] = cols["n_team"] * near_home
    for key in ("n_team", "n_dead", "n_mass", "n_ptr_near", "n_ptr_nearest", "n_agree"):
        x = cols[key]
        for half, tag in WINDOWS:
            cols["{}_{}".format(key, tag)] = _mean(x, -half, half)
        half, tag = SIDED
        cols["{}_before_{}".format(key, tag)] = _mean(x, -half, -1)
        cols["{}_after_{}".format(key, tag)] = _mean(x, 1, half)
    names = list(cols)
    return np.column_stack([np.asarray(cols[k], np.float32) for k in names]), names
