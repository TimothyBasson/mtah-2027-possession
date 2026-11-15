"""Per-provider raw table rows (speed, flight, gapdist) counted on raw and on folded labels."""
import glob, os, pickle, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external")]
import score_hmm as sh
from control_features import match_ball
PREP = os.path.join(ROOT, "logs_tune", "prep")
out = {}
for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz"))):
    n = os.path.basename(p)[:-4]
    d = sh.load(p); ball = match_ball(n, d)
    folded = pickle.load(open(os.path.join(PREP, n + ".pkl"), "rb"))["folded"]
    for view, lab in (("raw", d["labels"]), ("folded", folded)):
        dd = dict(d, labels=lab)
        out[n, view] = dict(speed=sh.speed_given_state([dd]), flight=sh.flight_given_state([dd], [ball]),
                            gap=sh.gap_distance_given_state([dd], [ball]), width=sh.width_given_state(dd))
    print(n, flush=True)
pickle.dump(out, open(os.path.join(PREP, "rows.pkl"), "wb"))
