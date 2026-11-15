"""One-off: per-match clue categories, labels and tables, pickled for fast decoder sweeps."""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "external")]
import hmm.decode as dec
import score_hmm as sh
import score_control as sc
from control import control_labels
from control_features import match_ball
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category

OUT = os.path.join(ROOT, "logs_tune", "prep")

def one(item):
    name, path = item
    d = sh.load(path)
    ball = match_ball(name, d)
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    r = dict(fps=fps, segments=[(int(a), int(b) + 1) for a, b in d["halves"]],
             labels=d["labels"], folded=control_labels(d),
             gapdist=gap_distance_category(ball_is_usable(bx, by, pitch), d_home, d_away).astype(np.int16),
             place=ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps).astype(np.int16),
             speed=sh.speed_cats(d).astype(np.int16), flight=sh.flight_cats(d, ball).astype(np.int16),
             width=sh.width_cats(d).astype(np.int16))
    pickle.dump(r, open(os.path.join(OUT, name + ".pkl"), "wb"))
    return name

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))}
    pickle.dump(sc.tables(paths), open(os.path.join(OUT, "tables.pkl"), "wb"))
    print("tables done", flush=True)
    with Pool(2) as pool:
        for n in pool.imap_unordered(one, list(paths.items())):
            print(n, flush=True)
