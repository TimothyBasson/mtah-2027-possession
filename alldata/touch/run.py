"""BAR-ATH labelled by an all-data model with the touch detector's evidence in the decode.

    python alldata/touch/run.py                                   # the submission's nets (with Tim's v8) + touches
    python alldata/touch/run.py --tags pwN pwNs1 --out alldata_learned_touch.csv
    python alldata/touch/run.py --tags xpwN xpwNs1 --extra --out alldata_touchin_touch.csv   # nets that read the touches too
    python alldata/touch/run.py --weight 0        # no evidence: must reproduce alldata/learned/run.py's file

Everything up to the probabilities is alldata/learned/run.py's own code (nothing there or in the
shipped files changes). The decode then adds, after each touch the detector (all-match net,
alldata/touch/models/t1.pt) sees on BAR-ATH's ball, weight x score to that team's log-probability
until the next touch or `hold` seconds. Weight 1 and hold 3 s were picked on validation matches
(tuse.py --kind val). No ground truth is read.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import tuse  # noqa: E402
import tbar  # noqa: E402
import textra  # noqa: E402

WEIGHT, HOLD, THR = 1.0, 3.0, 0.3


def learned_run():
    L = os.path.join(C.ROOT, "alldata", "learned")
    sys.path.insert(0, L)
    spec = importlib.util.spec_from_file_location("learned_run", os.path.join(L, "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["tpsN", "tpsNs1"])
    ap.add_argument("--out", default="alldata_learned_tim_touch.csv")
    ap.add_argument("--touch", default="t1")
    ap.add_argument("--weight", type=float, default=WEIGHT)
    ap.add_argument("--hold", type=float, default=HOLD)
    ap.add_argument("--extra", action="store_true", help="the nets also read the touch columns (textra.py)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    R = learned_run()
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    with_tim = R.needs_tim(a.tags)
    if with_tim:
        R.tim_ready()
    Fs, halves, n, names = R.features(with_tim=with_tim)
    q, _, _ = tbar.bar_probs(a.touch, "none")
    if a.extra:
        Fs = [np.concatenate([F, textra.columns(q[x:y + 1])[:len(F)]], 1) for F, (x, y) in zip(Fs, halves)]
        names = names + textra.NAMES
    probs = R.probabilities(a.tags, Fs, names)
    f, tm, sc = tuse.events(q, THR)
    ev, _ = tuse.evidence(n, f, tm, sc, a.weight, a.hold)
    codes = C.decode_ev(probs, halves, n, ev if a.weight else None, **R.DECODE)
    pred = np.select(
        [codes == C.H, codes == C.A, codes == C.DEAD], ["H", "A", "D"], "X")
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
