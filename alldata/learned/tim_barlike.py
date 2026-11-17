"""Tim's HMM v8, unchanged, on the outside matches with BAR-ATH-like balls (build.ball_version
1..): what his model gives when the ball is hidden and smoothed as BAR-ATH's is.

    python alldata/learned/tim_barlike.py features   # his classifier's features on those balls
    python alldata/learned/tim_barlike.py decode     # -> data/cache/alldata/tim/<match>.v<k>.npz

His classifier is refitted with each provider left out exactly as external/train_control.py
--loso does it (his rows(), fit() and cached features, so it learns from his own balls, as the
committed model that reads BAR-ATH did), then reads the left-out provider's matches on the
BAR-ATH-like balls. The decode is tim.run's with that ball. Nothing in hmm/ or external/ changes.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE]
sys.path += [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external")]
import build  # noqa: E402  (this folder's, before tim puts external/ first)
import tim  # noqa: E402

FEAT = os.path.join(ROOT, "data", "cache", "alldata", "tim_feat")
VERSIONS = (1, 2, 3)
STATE = {}


def paths():
    return {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}


def feature_job(job):
    name, v = job
    out = os.path.join(FEAT, f"{name}.v{v}.npz")
    if os.path.exists(out):
        return out
    from hmm.control import joint_features
    d = dict(np.load(paths()[name]))
    ball = build.ball_version(d, name, v).astype(float)
    players = d["players"].astype(np.float32)
    halves = [(int(a), int(b) + 1) for a, b in d["halves"]]
    args = (players[:, :, 0], players[:, :, 1], d["team"], ball[:, 0], ball[:, 1],
            tuple(float(x) for x in d["pitch"]), int(d["fps"]), halves)
    X, names, frames = joint_features(*args)
    Xs = joint_features(*args, swap=True)[0]
    os.makedirs(FEAT, exist_ok=True)
    np.savez_compressed(out, X=X, Xs=Xs, names=np.array(names), frames=frames)
    return out


def decode_job(job):
    name, v = job
    import hmm.decode as dec
    import score_hmm as sh
    from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category
    z = np.load(os.path.join(FEAT, f"{name}.v{v}.post.npz"))
    post, prior = z["post"], z["prior"]
    speed_t, flight_t, gap_t, width_t = STATE["tables"][name.split("_")[0]]
    d = sh.load(paths()[name])
    bxy = build.ball_version(d, name, v).astype(float)
    ball = np.column_stack([bxy, np.zeros(len(bxy))])
    fps, pitch = int(d["fps"]), tuple(float(x) for x in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0], ball[:, 1]
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t), (sh.flight_cats(d, ball), flight_t)]
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    ratio = np.log(np.clip(post, 1e-6, 1)) - np.log(prior)
    pred = dec.decode(clues, dec.transitions(), np.log(dec.START), segments, kickoffs=(), extra=extra,
                      direct=dec.CONTROL_WEIGHT * ratio,
                      min_possession=int(round(dec.MIN_POSSESSION_SECONDS * fps)))
    np.savez_compressed(os.path.join(tim.OUT, f"{name}.v{v}.npz"), labels=pred, post=post.astype(np.float32))
    return f"{name} v{v}"


def main():
    jobs = [(n, v) for n in paths() for v in VERSIONS]
    if sys.argv[1] == "features":
        with Pool(4) as pool:
            for r in pool.imap_unordered(feature_job, jobs):
                print(os.path.basename(r), flush=True)
        return
    import score_control as sc
    import train_control as tc
    from hmm.control import hold
    matches = tc.load_matches()
    for k in tc.SETS:
        todo = [(n, v) for n, v in jobs if n.startswith(k) and not os.path.exists(os.path.join(FEAT, f"{n}.v{v}.post.npz"))]
        if not todo:
            continue
        X, y, w = tc.rows(matches, [n for n in matches if not n.startswith(k)])
        model, prior = tc.fit(X, y, w), tc.class_shares(y, w)
        for n, v in todo:
            z = np.load(os.path.join(FEAT, f"{n}.v{v}.npz"))
            assert list(z["names"]) == matches[n][3]
            p = tc.posteriors(model, z["X"], z["Xs"]).astype(np.float32)
            np.savez_compressed(os.path.join(FEAT, f"{n}.v{v}.post.npz"), post=hold(p, matches[n][5]), prior=prior)
        print(k, "left out: classifier fitted and read", flush=True)
    STATE.update(tables=sc.tables(paths()))
    with Pool(4) as pool:
        for r in pool.imap_unordered(decode_job, jobs):
            print(r, flush=True)


if __name__ == "__main__":
    main()
