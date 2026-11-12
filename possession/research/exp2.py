import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
ms = load()
base = ["wid", "len", "vcx"]
allf = list(ms[0]["who"])
run(ms, who=(base, ["s1", "s3", "d2"]), tag="base, s1 s3 d2")
run(ms, who=(base, ["a1", "a3", "b1", "b3"]), tag="base, a1 a3 b1 b3")
run(ms, who=(base, ["s0.4", "a1", "a2", "a4", "b1", "b2", "b4"]), tag="base, s0.4 a1 a2 a4 b1 b2 b4")
run(ms, who=(base, ["s0.4", "a1", "a2", "a4", "a8", "b1", "b2", "b4", "b8"]), tag="base, + a8 b8")
run(ms, who=(base + ["speed"], ["s0.4", "a1", "a2", "a4", "b1", "b2", "b4"]), tag="base+speed, 7 views")
run(ms, who=(allf, ["s0.4", "a1", "a2", "a4", "b1", "b2", "b4"]), tag="all, 7 views")
run(ms, who=(allf, ["s0.4", "a1", "a2", "a4", "a8", "b1", "b2", "b4", "b8"]), tag="all, 9 views")
