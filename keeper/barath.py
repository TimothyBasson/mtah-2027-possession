"""Cache BAR-ATH's raw tracking as arrays (px, py, ball, per-frame team, keeper slots, halves)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hmm.decode as dec
from hmm.clues import read_positions, team_by_frame, keeper_slots
from starter.baseline import team_of_each_track

OUT = os.path.join(ROOT, "data", "cache", "keeper_barath.npz")


def load():
    if os.path.exists(OUT):
        return dict(np.load(OUT))
    meta, raw = dec.require(["atd_meta", "atd_raw"])
    tot = team_of_each_track(meta)
    n = dec.source_frames("tactical")
    pitch = dec.GAMES[dec.GAME]["pitch"]
    px, py, bx, by = read_positions(raw, len(tot), n, pitch)[:4]
    keepers = keeper_slots(meta)
    halves = dec.graded_segments()
    team = team_by_frame(px, tot, keepers, halves, pitch[0])
    out = dict(px=px, py=py, bx=bx, by=by, team=team, team_of_track=np.asarray(tot),
               keepers=np.array(keepers), halves=np.array(halves), pitch=np.array(pitch, float),
               fps=dec.FPS)
    np.savez(OUT, **out)
    return out


if __name__ == "__main__":
    d = load()
    print({k: getattr(v, "shape", v) for k, v in d.items()})
