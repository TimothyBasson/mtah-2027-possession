"""BAR-ATH file: the submission with the kink gate on top.

    python alldata/briefsafe/run.py                       # -> evaluation/results/alldata_briefsafe_touch.csv
    python alldata/briefsafe/run.py --fine pcrN pcrNs1 --m 0.1

Base: the submission's probabilities (alldata/learned/run.py, nets tpsN + tpsNs1 with Tim's v8 as
input, keeper fix on load); they are rebuilt first if not cached. Fine: the contest nets trained to
cut brief possessions (alldata/contest, cfit.py --fold none). Where the two name different teams,
the fine net's team is kept only if the last kink of BAR-ATH's ball (where it was really played,
alldata/contest/cfeats.py) was nearer that team by at least --m; otherwise the submission's
probabilities stand. Then the submission's decode with the touch evidence. No ground truth is read.
"""
import argparse, importlib.util, os, subprocess, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", default=["tpsN", "tpsNs1"])
    ap.add_argument("--fine", nargs="+", default=["kbN", "kbNs1"])
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--out", default="alldata_briefsafe_touch.csv")
    a = ap.parse_args()
    torch.set_num_threads(1)
    C = mod("contest_run", os.path.join(ROOT, "alldata", "contest", "run.py"))
    R, T = C.R, C.T
    fb = os.path.join(T.CACHE, "bar_ath", "+".join(a.base) + ".npz")
    if not os.path.exists(fb):
        subprocess.run([sys.executable, os.path.join(ROOT, "alldata", "learned", "run.py"), "--tags", *a.base, "--out", "_base.csv"],
                       cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED="0"), check=True)
    zb = np.load(fb)
    base = [zb[f"p{i}"] for i in range(len(zb.files))]
    Fs, halves, n, names = C.features()            # BAR-ATH rows + contest columns, real ball (keeper fix on load)
    fine = [0] * len(Fs)
    for tag in a.fine:
        f = next((p for p in (os.path.join(HERE, "models", tag + ".pt"), os.path.join(ROOT, "alldata", "contest", "models", tag + ".pt")) if os.path.exists(p)), "")
        ck = torch.load(f if os.path.exists(f) else os.path.join(T.CACHE, "runs", tag, "none", "net.pt"), map_location="cpu")
        cols = [names.index(c) for c in ck["names"]]
        net = T.Net(len(cols), ck["d"]); net.load_state_dict(ck["sd"])
        sym = C.cfeats.symmetry(ck["names"])
        for i, F in enumerate(Fs):
            fine[i] = fine[i] + R.head(T.predict(net, F[:, cols], sym)) / len(a.fine)
    km = names.index("k_last_m")
    probs, sent, dis = [], 0, 0
    for pb, pf, F in zip(base, fine, Fs):
        assert len(pb) == len(pf) == len(F)
        tb, tf = pb.argmax(1), pf.argmax(1)
        diff = (tb != tf) & (tf < 2)
        ok = F[:, km] * np.where(tf == 0, 1.0, -1.0) >= a.m
        use_base = diff & ~ok
        p = pf.copy(); p[use_base] = pb[use_base]
        probs.append(p); sent += use_base.sum(); dis += diff.sum()
    print(f"rows where base and fine disagree {dis} of {sum(len(p) for p in probs)}; fine kept on {dis - sent}")
    os.makedirs(os.path.join(T.CACHE, "bar_ath"), exist_ok=True)
    np.savez_compressed(os.path.join(T.CACHE, "bar_ath", "briefsafe.npz"), **{f"p{i}": p for i, p in enumerate(probs)})
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    sys.path.insert(0, os.path.join(ROOT, "alldata", "touch"))
    touch_run = mod("touch_run", os.path.join(ROOT, "alldata", "touch", "run.py"))
    pred = touch_run.labels_with_touches(probs, halves, n, R.DECODE)
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
