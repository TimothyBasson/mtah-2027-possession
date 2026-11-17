"""Label BAR-ATH with the contest nets (cfit.py --fold none) and write a submission file.

    python alldata/contest/run.py --features                      # BAR-ATH's rows and contest columns (before cfit.py --fold none)
    python alldata/contest/run.py                                  # pcrN + pcrNs1 with the touch evidence -> alldata_contest_touch.csv
    python alldata/contest/run.py --tags pcN pcNs1 --out alldata_contest_inputs_touch.csv

Same decode, touch evidence (alldata/touch/run.py) and layout as alldata/learned/run.py;
independent of hmm/. No ground truth is read.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "alldata", "learned")]
import cfeats  # noqa: E402

spec = importlib.util.spec_from_file_location("learned_run", os.path.join(ROOT, "alldata", "learned", "run.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
T = R.T


def features():
    import load
    Fs, halves, n, names = R.features(with_tim=False)
    players, ball, team, _ = load.load()
    pxy, bxy = players[:, :, :2].astype(np.float64), ball[:, :2].astype(np.float64)
    Cs = [cfeats.half_columns(pxy[a:b + 1], bxy[a:b + 1], team, load.FPS) for a, b in halves]
    cnames = list(cfeats.NAMES)
    save = {}
    for i, F in enumerate(Fs):
        save.update({f"F0_{i}": F, f"fr_{i}": np.arange(len(F)), f"y_{i}": np.full(len(F), -1, np.int8), f"yf_{i}": np.full(len(F), -1, np.int8)})
    for sub, nm, d in (("bar", names, save), ("contest_bar", cnames, {f"F0_{i}": C for i, C in enumerate(Cs)})):
        os.makedirs(os.path.join(T.CACHE, sub), exist_ok=True)
        np.savez(os.path.join(T.CACHE, sub, "BAR-ATH.npz"), names=np.array(nm), nhalves=len(halves), **d)
    return [np.concatenate([F, C], 1) for F, C in zip(Fs, Cs)], halves, n, names + cnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pcrN", "pcrNs1"])
    ap.add_argument("--out", default="alldata_contest_touch.csv")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--no-touch", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    Fs, halves, n, names = features()
    if a.features:
        return
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    probs = [0] * len(Fs)
    for tag in a.tags:
        f = os.path.join(HERE, "models", tag + ".pt")
        ck = torch.load(f if os.path.exists(f) else os.path.join(T.CACHE, "runs", tag, "none", "net.pt"), map_location="cpu")
        cols = [names.index(c) for c in ck["names"]]
        net = T.Net(len(cols), ck["d"])
        net.load_state_dict(ck["sd"])
        sym = cfeats.symmetry(ck["names"])
        for i, F in enumerate(Fs):
            probs[i] = probs[i] + R.head(T.predict(net, F[:, cols], sym)) / len(a.tags)
    os.makedirs(os.path.join(T.CACHE, "bar_ath"), exist_ok=True)
    np.savez_compressed(os.path.join(T.CACHE, "bar_ath", "+".join(a.tags) + ".npz"), **{f"p{i}": p for i, p in enumerate(probs)})
    if a.no_touch:
        pred = R.decode.labels_from(probs, halves, n, **R.DECODE)
    else:
        sys.path.insert(0, os.path.join(ROOT, "alldata", "touch"))
        spec = importlib.util.spec_from_file_location("touch_run", os.path.join(ROOT, "alldata", "touch", "run.py"))
        touch_run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(touch_run)
        pred = touch_run.labels_with_touches(probs, halves, n, R.DECODE)
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
