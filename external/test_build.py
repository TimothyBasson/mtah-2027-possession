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
