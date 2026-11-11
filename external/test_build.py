"""uv run python external/test_build.py"""

from build import metrica_labels

ev = [  # (team, type, subtype, start, end)
    (0, "SET PIECE", "KICK OFF", 2, 2),
    (0, "PASS", "", 2, 4),
    (1, "CHALLENGE", "TACKLE-LOST", 5, 5),   # a challenge alone doesn't move possession
    (1, "RECOVERY", "INTERCEPTION", 6, 6),
    (1, "BALL OUT", "", 7, 8),               # dead from the ball-out's end frame
    (0, "RECOVERY", "", 11, 11),             # goal kick logged as a recovery restarts play
]
got = "".join(metrica_labels(ev, {f: f for f in range(13)}, [(1, 12)]))
assert got == "XDHHHHAADDDHH", got
print("ok")

from build import resample, skillcorner_labels
import numpy as np

g = np.array([-1, 0, -1, -1, 1, -1, -1, -1])   # who SkillCorner names per frame
alive = np.array([0, 1, 1, 1, 1, 1, 0, 1], bool)
team = np.array([0, 0, 0, 0, 0, 0, 0, 1])       # the phase's team, used until a group is named
got = "".join(skillcorner_labels(g, alive, team))
assert got == "DHHHAADA", got
x = resample(np.array([0., 10., 20., np.nan], np.float32), 8)  # 10 fps -> 25 fps
assert np.allclose(x[:6], [0, 4, 8, 12, 16, 20]) and np.isnan(x[6:]).all(), x  # 20 sits on a gap's edge
print("ok")
