"""BAR-ATH labelled by the submission's nets with brief touches written into the decode.
An optional second file; the submission (alldata/learned/run.py) is not changed.

    python alldata/shortdec/apply.py [--tau 0.5] [--out alldata_learned_tim_touches.csv]

Same nets, features and plain decode as alldata/learned/run.py; then dec.insert_touches on the
head taught on the labels as built. Reads no ground truth.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import dec  # noqa: E402

spec = importlib.util.spec_from_file_location("learned_run", os.path.join(ROOT, "alldata", "learned", "run.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["tpsN", "tpsNs1"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default="alldata_learned_tim_touches.csv")
    a = ap.parse_args()
    R.torch.set_num_threads(1)
    from starter.baseline import RESULTS_DIR, write_submission
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    with_tim = R.needs_tim(a.tags)
    if with_tim:
        R.tim_ready()
    Fs, halves, n, names = R.features(with_tim=with_tim)
    R.WHICH = "both"
    both = R.probabilities(a.tags, Fs, names)
    R.WHICH = "built"
    built = R.probabilities(a.tags, Fs, names)
    pred = np.full(n, "X", "<U1")
    added = 0
    for p, pb, (s, e) in zip(both, built, halves):
        m = e - s + 1
        P, Pe = dec.upsample(p, m), dec.upsample(pb, m)
        plain = dec.decode_half(P, **R.DECODE)
        path = dec.decode_insert(P, Pe, tau=a.tau, **R.DECODE)
        added += int((np.diff(path) != 0).sum() - (np.diff(plain) != 0).sum()) // 2
        pred[s:e + 1] = R.decode.LETTERS[path]
    for (s, _), kick in zip(halves, KICKOFF_VIDEO_FRAMES.values()):
        pred[s:kick] = "D"
    print(f"{added} brief touches written in")
    print("wrote", write_submission(os.path.join(RESULTS_DIR, a.out), pred))


if __name__ == "__main__":
    main()
