"""The touch detector's evidence in the BAR-ATH decode; alldata/learned/run.py calls it.

    python alldata/learned/run.py                 # the submission: its nets + this evidence
    python alldata/learned/run.py --no-touch      # without it: reproduces alldata_learned_tim.csv byte for byte

After each touch the detector (the all-match net, alldata/touch/models/t1.pt) sees on BAR-ATH's
ball, weight x score is added to that team's log-probability until the next touch or `hold`
seconds. Weight 1 and hold 3 s were picked on validation matches (tuse.py --kind val). No ground
truth is read.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import tuse  # noqa: E402
import tbar  # noqa: E402

TOUCH, WEIGHT, HOLD, THR = "t1", 1.0, 3.0, 0.3


def labels_with_touches(probs, halves, n, dec, touch=TOUCH, weight=WEIGHT, hold=HOLD):
    """decode.labels_from(probs, halves, n, **dec) with the touch evidence; H/A/D/X letters."""
    torch.set_num_threads(1)
    q, _, _ = tbar.bar_probs(touch, "none")
    f, tm, sc = tuse.events(q, THR)
    ev, _ = tuse.evidence(n, f, tm, sc, weight, hold)
    codes = C.decode_ev(probs, halves, n, ev if weight else None, **dec)
    return np.select([codes == C.H, codes == C.A, codes == C.DEAD], ["H", "A", "D"], "X")
