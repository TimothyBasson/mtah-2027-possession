"""How much of the knock-on loss a touch signal can remove, before any detector is built.

    python alldata/touch/oracle.py [--tag pw1e]

Touches here come from the ball AS RECORDED (hmm.ball.detect_touches, the detector behind the
folded labels): the best any detector reading the smoothed ball could hope to recover. Prints the
base, then the decode with touch evidence at several strengths.
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from hmm.ball import detect_touches  # noqa: E402


def raw_touches(name, v=0):
    """Touches of the recorded ball (v=0) or, for comparison, the same hand-set detector run on
    the BAR-ATH-like ball of version v."""
    d = dict(np.load(C.match_path(name)))
    ball = d["ball"].astype(float)
    if v:
        import tfeats
        ball = tfeats.learned_build().ball_version(d, name, v).astype(float)
    dh, da = C.player_dists(d, ball[:, :2])
    t = detect_touches(ball[:, 0], ball[:, 1], dh, da, int(d["fps"]))
    return t["frame"], t["team"]


def evidence(n, frames, teams, weight, hold, scores=None):
    """(n, 3) evidence and the kept frames: after a touch by a team, that team's log-probability
    is raised by weight (times the touch's score) until the next touch or `hold` seconds."""
    ev = np.zeros((n, 3))
    keep = np.zeros(n, bool)
    ok = teams >= 0
    frames, teams = frames[ok], teams[ok]
    scores = np.ones(len(frames)) if scores is None else scores[ok]
    for j, (f, k) in enumerate(zip(frames, teams)):
        e = min(f + int(hold * 25), frames[j + 1] if j + 1 < len(frames) else n, n)
        ev[f:e, k] += weight * scores[j]
        keep[f] = True
    return ev, keep


def one(args):
    name, tag, settings, v = args
    raw, fold, spans = C.truths(name)
    n = len(raw)
    fr, tm = raw_touches(name, v)
    pr, pf = C.probs(tag, name, "s"), C.probs(tag, name, "s", "f")
    out = {}
    base_f = C.decode_ev(pf, spans, n, **C.FOLDED)
    out["base"] = C.evaluate(name, C.decode_ev(pr, spans, n, **C.RAW), base_f)
    for w, hold, protect in settings:
        ev, keep = evidence(n, fr, tm, w, hold)
        p = C.decode_ev(pr, spans, n, ev, keep=keep if protect else None, **C.RAW)
        pfo = C.decode_ev(pf, spans, n, ev, keep=keep if protect else None, **C.FOLDED)
        out[(w, hold, protect)] = C.evaluate(name, p, pfo)
    # the touch timeline alone inside the model's live frames
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pw1e")
    ap.add_argument("--v", type=int, default=0, help="0: recorded ball (the ceiling); 1 or 4: the hand-set detector on the smoothed ball")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    settings = [(w, h, pr) for w in (2, 4) for h in (1, 2) for pr in (0, 1)]
    res = dict(C.pool_map(one, [(n, a.tag, settings, a.v) for _, n in C.test_matches(a.tag)], a.workers))
    base = {n: r["base"] for n, r in res.items()}
    C.report("base", base)
    for s in settings:
        C.report(f"raw touches w{s[0]} hold{s[1]} keep{s[2]}", {n: r[s] for n, r in res.items()}, base)


if __name__ == "__main__":
    main()
