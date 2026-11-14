"""Cache per match: v6 log-emissions, raw and folded labels, segments."""
import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, ROOT + "/evaluation", ROOT + "/external"]
import hmm.decode as dec
import score_hmm as sh
import score_control as sc
from control import control_labels
from control_features import match_ball
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category
from hmm.model import build_log_emit
from multiprocessing import Pool
OUT = ROOT + "/data/cache/wip/emit"
os.makedirs(OUT, exist_ok=True)
paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(ROOT + "/data/external/*.npz"))}
TABS = None
def run(name):
    k = name.split("_")[0]
    speed_t, flight_t, gap_t, width_t = TABS[k]
    d = sh.load(paths[name]); ball = match_ball(name, d)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t), (sh.flight_cats(d, ball), flight_t)]
    emit = build_log_emit([(o, t, dec.WEIGHT) for o, t in clues] + [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)])
    np.savez_compressed(f"{OUT}/{name}.npz", emit=emit.astype(np.float32), labels=d["labels"],
                        folded=control_labels(d), halves=d["halves"], fps=fps)
    return name
if __name__ == "__main__":
    TABS = sc.tables(paths)
    with Pool(4) as p:
        for n in p.imap_unordered(run, list(paths)): print(n, flush=True)
