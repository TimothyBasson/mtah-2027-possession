"""The control rule on its own, no learning: what the touches and the gaps say by themselves.

    python alldata/physics/rule.py            # writes data/cache/alldata/pred/rule*/, then score.py

Team: the team of the last touch, but a lone touch between two touches of the other team does
not take the ball (CHALLENGE.md: "a simple touch without control" is not a change); a team takes
it with two touches in a row. Dead: the ball out of view for more than GAP seconds, or beyond a
line. Variants: "rule" (touches only), "rule_prior" (the players-only model where the ball is
hidden, and its dead-ball call throughout).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from build import table_path  # noqa: E402
from train import save_probs  # noqa: E402

GAP = 4.0
SOFT = 0.9


def rule_probs(t, prior=False):
    F, names = t["F"], list(t["names"])
    g = lambda k: F[:, names.index(k)]
    controlled = (g("t_run_back") + g("t_run_fwd")) >= 2
    team = np.where(controlled, g("t_last1_team"), g("t_last2_team"))
    team = np.where(team == 0, g("t_last1_team"), team)
    hidden = g("g_shown") < 0.5
    out = np.nan_to_num(np.minimum(g("b_side_margin"), g("b_end_margin")), nan=1.0) < -0.5
    dead = (hidden & (g("g_len") > GAP)) | out
    p = np.full((len(F), 3), (1 - SOFT) / 2)
    p[team > 0, 0] = SOFT
    p[team < 0, 1] = SOFT
    p[team == 0] = [0.45, 0.45, 0.1]
    p[dead] = [(1 - SOFT) / 2, (1 - SOFT) / 2, SOFT]
    if prior:
        q = np.stack([g("q_home"), g("q_away"), g("q_dead")], 1)
        p[hidden] = q[hidden]
        live = ~hidden & ~out
        p[live] = p[live] * (1 - q[live, 2:3])
        p[live, 2] = q[live, 2]
    return p / p.sum(1, keepdims=True)


def main():
    for n in C.names():
        z = np.load(table_path(n, "barath", "f5"))
        t = {k: z[k] for k in z.files}
        save_probs("rule", n, t, rule_probs(t))
        save_probs("rule_prior", n, t, rule_probs(t, True))
        q = np.stack([t["F"][:, list(t["names"]).index(k)] for k in ("q_home", "q_away", "q_dead")], 1)
        save_probs("prior", n, t, q)


if __name__ == "__main__":
    main()
