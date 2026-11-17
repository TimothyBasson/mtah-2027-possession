"""Keeper categories around each BAR-ATH keeper note, per mode."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hmm import decode as dec, keeper
bx, by, usable, px, py, ks, pitch, halves = dec.keeper_positions()
notes = json.load(open(os.path.join(ROOT, "data/viz/notes/BAR-ATH.json")))["notes"]
keep_notes = [23946, 26688, 69326, 76462, 82248, 123481, 133928, 168311, 69725, 156799, 45 * 0]
for mode in ("next", "last", "both", "box"):
    c = keeper.keeper_category(bx, by, usable, px, py, ks, pitch, 25, halves, ends=mode)
    g = np.zeros(len(c), bool)
    for lo, hi in halves:
        g[lo:hi] = True
    print(mode, "frames per cat (graded):", np.bincount(c[g & (c >= 0)], minlength=4))
    for n in notes:
        f = n["frame"]
        if n["tag"] == "False stoppage" or "keeper" in n["text"].lower():
            w = c[f - 250:f + 500]
            print("   %6d %s %-15s miss H/A %4d %4d  still H/A %4d %4d" % (
                f, "%d:%04.1f" % divmod(f / 25, 60), n["tag"][:15],
                *(np.sum(w == k) for k in range(4))))
