"""Tracking around each BAR-ATH dead-ball note: ball seen, where, nearest keeper, player speed."""
import json, os, sys, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hmm import decode as dec
from hmm.clues import read_positions, keeper_slots, median_player_speed
from starter.baseline import team_of_each_track
from games import GAMES
meta, raw = dec.require(["atd_meta", "atd_raw"], dec.GAME, None)
n = dec.source_frames("tactical", dec.GAME)
pitch = GAMES[dec.GAME]["pitch"]
tt = team_of_each_track(meta); kp = keeper_slots(meta)
cache = os.path.join(ROOT, "data/cache/barath_pos.npz")
if os.path.exists(cache):
    z = np.load(cache); px, py, bx, by = z["px"], z["py"], z["bx"], z["by"]
else:
    px, py, bx, by = read_positions(raw, len(tt), n, pitch)[:4]
    np.savez(cache, px=px, py=py, bx=bx, by=by)
spd = median_player_speed(px, py, 25)
lab = np.array([r["label"] for r in csv.DictReader(open("/mnt/project-files/v9/hmm_v9.csv"))])
print("pitch", pitch, "keepers", kp)
for nt in json.load(open(os.path.join(ROOT, "data/viz/notes/BAR-ATH.json")))["notes"]:
    f = nt["frame"]
    print(f, nt["tag"], "|", nt["text"][:50])
    for t in range(-4, 12, 2):
        g = f + t * 25
        kd = [np.hypot(px[g, k] - bx[g], py[g, k] - by[g]) for k in kp]
        print("   %+3ds %s ball=%s spd=%.2f keeperd=%s" % (t, lab[g],
              "-" if np.isnan(bx[g]) else "(%.0f,%.0f)" % (bx[g], by[g]), spd[g],
              ",".join("%.1f" % v for v in kd)))
