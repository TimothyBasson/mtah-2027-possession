"""Tim's decoder (hmm/, v8) as one more input, without changing any of his files.

    python alldata/physics/tim.py sim        # the 30 matches as BAR-ATH would show them
    python alldata/physics/tim.py features   # his classifier's features on them (external/control_features.py)
    python alldata/physics/tim.py train      # his classifier, held out by provider and on all 30
    python alldata/physics/tim.py decode     # his decoder's labels for the 30, and its scores
    python alldata/physics/tim.py bar_ath    # his decoder on BAR-ATH with the all-30 classifier

His scripts score the outside matches with a ball knocked out at random (--seen 0.59). Here they
are run on the same simulated matches this folder trains on (common.shown_ball mode "barath":
smoothed ball, long gaps, a dead ball in view in some stoppages; players lost as BAR-ATH loses
them), so that his labels on the outside matches are as good, and as bad, as they will be on
BAR-ATH, and the classifier that reads them learns how far to trust them. His classifier is
retrained on these simulated matches for the same reason, into this folder's cache; hmm/models/
is not touched.

Outputs under data/cache/alldata/tim/: external/<match>.npz (simulated matches),
control_v8_<match>.npz (his features), control_loso.npz, control_state.npz, labels/<match>.npy.
"""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

TIM = os.path.join(C.CACHE, "tim")
os.environ["CONTROL_CACHE"] = TIM          # external/control_features.py's cache folder
sys.path[:0] = [C.ROOT, os.path.join(C.ROOT, "external")]
WEIGHT = 0.2                               # hmm.decode.CONTROL_WEIGHT, v8's default
CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def sim_path(name):
    return os.path.join(TIM, "external", name + ".npz")


def sim_one(name):
    out = sim_path(name)
    if not os.path.exists(out):
        d = C.load_match(name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, players=C.shown_players(d).astype(np.float32),
                            ball=C.shown_ball(d, "barath").astype(np.float32), team=d["team"],
                            labels=d["labels"], halves=np.array(d["halves"]), pitch=np.array(d["pitch"]),
                            fps=d["fps"])
    return name


def feat_one(name):
    import control_features as CF
    CF.match_ball = lambda n, d, seen=None: d["ball"]   # the simulated ball as it is
    CF.cache_path = lambda n: os.path.join(TIM, "control_v8_{}.npz".format(n))
    return CF.build(sim_path(name))


def decode_one(name):
    import hmm.decode as dec
    import score_hmm as sh
    from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category
    d = sh.load(sim_path(name))
    loso = np.load(os.path.join(TIM, "control_loso.npz"))
    ball = d["ball"]
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    segments = [(int(a), int(b) + 1) for a, b in d["halves"]]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    # the decoder's shipped tables throughout, as on BAR-ATH
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), dec.gap_distance_table()),
             (ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps), dec.place_table()),
             (sh.speed_cats(d), dec.speed_table()), (sh.flight_cats(d, ball), dec.flight_table())]
    extra = [(sh.width_cats(d), dec.width_table(), dec.WIDTH_WEIGHT)]
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - np.log(loso["prior_" + name])
    pred = dec.decode(clues, dec.transitions(), np.log(dec.START), segments, kickoffs=(), extra=extra,
                      direct=WEIGHT * ratio)
    pred[d["labels"] == "X"] = "X"
    os.makedirs(os.path.join(TIM, "labels"), exist_ok=True)
    np.save(os.path.join(TIM, "labels", name + ".npy"), np.array([CODES[c] for c in pred], np.int8))
    import possession_metrics as pm
    z = np.load(os.path.join(C.CACHE, "f5", "barath", name + ".npz"))  # labels as given and folded
    p = pm.label_codes(pred)
    return name, [pm.evaluate(pm.label_codes(z[k]), p, bootstrap_n=0)["spell_f1_mean"] for k in ("labels", "folded")]


def labels(name):
    """Tim's label per frame for a match, as codes 0 H, 1 A, 2 D, 3 X."""
    if name == "bar_ath":
        import pandas as pd
        lab = pd.read_csv(os.path.join(TIM, "hmm_v8_matched.csv")).label.values
        return np.array([CODES[c] for c in lab], np.int8)
    return np.load(os.path.join(TIM, "labels", name + ".npy"))


def features(lab, frames, halves, fps=25):
    """Columns for the classifier from Tim's labels: his call now and a few seconds either way,
    how long since and until he changes it. Returns (columns, columns with the teams exchanged)."""
    out, names = [], []
    n = len(lab)
    team = np.where(lab == 0, 1.0, np.where(lab == 1, -1.0, 0.0))
    dead = (lab == 2).astype(float)
    chg = np.r_[True, lab[1:] != lab[:-1]]
    for a, b in halves:
        chg[a] = True
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(chg, idx, 0))
    nxt = np.minimum.accumulate(np.where(np.r_[chg[1:], True], idx + 1, n)[::-1])[::-1]
    prev_team = team[np.clip(last - 1, 0, n - 1)]
    next_team = team[np.clip(nxt, 0, n - 1)]
    cols = {"z_team": team[frames], "z_dead": dead[frames],
            "z_since": np.minimum((frames - last[frames]) / fps, 60), "z_until": np.minimum((nxt[frames] - frames) / fps, 60),
            "z_prev_team": prev_team[frames], "z_next_team": next_team[frames],
            "z_prev_dead": dead[np.clip(last - 1, 0, n - 1)][frames], "z_next_dead": dead[np.clip(nxt, 0, n - 1)][frames]}
    for o in (-3.0, -1.0, 1.0, 3.0):
        j = np.clip(frames + int(o * fps), 0, n - 1)
        cols[f"z_team@{o:+g}"] = team[j]
        cols[f"z_dead@{o:+g}"] = dead[j]
    names = list(cols)
    F = np.stack([cols[k] for k in names], 1).astype(np.float32)
    Fs = F.copy()
    for i, k in enumerate(names):
        if "team" in k:
            Fs[:, i] *= -1
    return F, Fs, names


def main():
    step = sys.argv[1]
    names = C.names()
    if step == "sim":
        with ProcessPoolExecutor(3) as ex:
            for n in ex.map(sim_one, names):
                print("simulated", n, flush=True)
    elif step == "features":
        with ProcessPoolExecutor(3) as ex:
            for n in ex.map(feat_one, names):
                print("features", n, flush=True)
    elif step == "train":
        py, script = sys.executable, os.path.join(C.ROOT, "external", "train_control.py")
        subprocess.run([py, script, "--loso"], check=True, env=os.environ)
        subprocess.run([py, script, "--out", os.path.join(TIM, "control_state.npz")], check=True, env=os.environ)
    elif step == "decode":
        with ProcessPoolExecutor(3) as ex:
            res = dict(ex.map(decode_one, names))
        for k, title in ((0, "as given"), (1, "folded")):
            per = [np.mean([res[n][k] for n in names if C.provider(n) == p]) for p in C.PROVIDERS]
            print(f"Tim's v8 on the simulated matches, {title}: " + " / ".join(f"{x:.3f}" for x in per)
                  + f"   mean {np.mean(per):.3f}")
    elif step == "bar_ath":
        out = os.path.join(TIM, "hmm_v8_matched.csv")
        subprocess.run([sys.executable, "-m", "hmm.decode", "--control-model", os.path.join(TIM, "control_state.npz"),
                        "--out", out], check=True, cwd=C.ROOT)


if __name__ == "__main__":
    main()
