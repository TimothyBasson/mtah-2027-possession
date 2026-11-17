"""The submission pipeline (alldata/learned/run.py) on BAR-ATH's ball with the tracker's gaps
filled from our own detector on the tactical video. Main's submission is not changed.

    python alldata/video/run.py                       # all gaps -> evaluation/results/alldata_video_all.csv
    python alldata/video/run.py --max-gap 250 --out alldata_video_g250.csv   # only gaps up to 10 s

Steps: targets.py lists the frames to scan, detect.py runs YOLO11s (COCO "sports ball", stock
weights) on them (needs the tactical video, cut or whole; its detections are cached under
data/cache/video/), ballfill.py picks and joins them. The learned nets, Tim's v8 input (still on the
tracker's ball: his code and cache are not touched) and the touch decode then run unchanged, with
their caches for this ball kept apart (data/cache/alldata/video_<tag>/)."""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "alldata", "learned"), os.path.join(ROOT, "possession")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gap", type=int, default=None, help="fill only tracker gaps up to this many frames")
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--out", default=None)
    a, rest = ap.parse_known_args()
    import ballfill, load
    tag = f"g{a.max_gap or 'all'}" + (f"_c{a.min_conf}" if a.min_conf else "")
    out = a.out or f"alldata_video_{tag}.csv"
    raw = load.load_raw
    players, ball, team, halves = raw()
    filled_ball, filled = ballfill.fill(ball, halves, a.max_gap, a.min_conf or ballfill.MIN_CONF)
    graded = np.zeros(len(ball), bool)
    for s, e in halves:
        graded[s:e + 1] = True
    print(f"ball seen on {np.isfinite(ball[graded, 0]).mean():.3f} of graded frames, "
          f"{np.isfinite(filled_ball[graded, 0]).mean():.3f} with the video fill ({filled.sum()} frames filled)")
    load.load_raw = lambda: (players, filled_ball, team, halves)

    import importlib.util
    spec = importlib.util.spec_from_file_location("learned_run", os.path.join(ROOT, "alldata", "learned", "run.py"))
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    R.tim_ready()  # Tim's v8 on the tracker's ball, as in the submission (made once, then cached)
    R.tim_ready = lambda: None
    R.T.CACHE = os.path.join(ROOT, "data", "cache", "alldata", f"video_{tag}")
    sys.path.insert(0, os.path.join(ROOT, "alldata", "touch"))
    import common as C
    C.TOUCH = os.path.join(R.T.CACHE, "touch")
    sys.argv = [sys.argv[0], "--out", out] + rest
    R.main()


if __name__ == "__main__":
    main()
