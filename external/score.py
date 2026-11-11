"""Score the possession model on the out-of-sample matches from build.py.

    uv run python external/score.py [--model DIR] [--seen 0.59] [param=value ...]

--model   folder holding the model (model.py, features.py, load.py); default ../possession
--seen    knock the ball out in random runs until it is seen this share of frames, like our ATD
          (59% of graded frames). IDSSE's optical tracking sees it ~100%, which flatters the model.
          Seeded, so a rerun gives the same score.

The model reads no set-piece tags here: these matches have no Smart Tagging.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def drop_ball(ball, seen, fps, seed=0, labels=None):
    """Two-state on/off runs with ~2 s mean gaps, so the ball vanishes in stretches, not frames.
    Only tops up what is already missing: a match at or below `seen` is left alone. With labels,
    `seen` is measured on labelled frames: SkillCorner loses the ball in its unlabelled cutaways."""
    graded = slice(None) if labels is None else labels != "X"
    seen = seen / np.isfinite(ball[graded, 0]).mean()
    if seen >= 1:
        return ball
    rng = np.random.default_rng(seed)
    off_len = 2 * fps
    on_len = off_len * seen / (1 - seen)
    keep = np.ones(len(ball), bool)
    i, on = 0, True
    while i < len(ball):
        n = 1 + int(rng.exponential(on_len if on else off_len))
        keep[i:i + n] = on
        i, on = i + n, not on
    out = ball.copy()
    out[~keep] = np.nan
    return out


def frames(players, ball, team, features):
    """The dict features.compute() builds for our match, from any match's arrays."""
    dh, da = features.nearest_by_team(players, ball, team)
    seen = np.isfinite(ball[:, 0])
    L, W = features.PITCH
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        out = seen & ((ball[:, 0] < -0.5) | (ball[:, 0] > L + 0.5)
                      | (ball[:, 1] < -0.5) | (ball[:, 1] > W + 0.5))
        mean_speed = np.nanmean(players[:, :, 2], 1)
    return dict(players=players, ball=ball, team=team, dh=dh, da=da, seen=seen, out=out,
                mean_speed=mean_speed, n_trk=np.isfinite(players[:, :, 0]).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "possession"))
    ap.add_argument("--seen", type=float, default=1.0)
    ap.add_argument("params", nargs="*")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.model))
    import features
    import model
    params = {k: (v if v.isalpha() else float(v)) for k, v in (p.split("=") for p in a.params)}

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz"))):
        d = np.load(path)
        assert int(d["fps"]) == model.FPS, f"{path}: model assumes {model.FPS} fps"
        halves = [tuple(h) for h in d["halves"]]
        mask = np.zeros(len(d["labels"]), bool)
        for s, e in halves:
            mask[s:e + 1] = True
        # the model reads these as module constants of our match; point them at this one
        features.PITCH = model.PITCH = tuple(d["pitch"])
        model.HALVES = halves
        model.graded_mask = lambda n, m=mask: m
        F = frames(d["players"], drop_ball(d["ball"], a.seen, model.FPS, labels=d["labels"]), d["team"], features)
        pred = model.predict(F, params)
        r = pm.evaluate(pm.label_codes(d["labels"]), pm.label_codes(pred), bootstrap_n=0)
        name = os.path.basename(path)[:-4]
        rows.append((name, r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"],
                     np.isfinite(F["ball"][mask, 0]).mean()))
        print(f"{name:14s} spellF1 {rows[-1][1]:.3f}  macroF1 {rows[-1][2]:.3f}  alive/dead {rows[-1][3]:.3f}"
              f"  team {rows[-1][4]:.3f}  ball seen {rows[-1][5]:.0%}", flush=True)
    # per set too, so a mean over all sets isn't compared with one taken before a set was added
    for p in sorted({r[0].split("_")[0] for r in rows}) + [""]:
        sel = [r[1:] for r in rows if r[0].startswith(p)]
        m = np.mean(sel, 0)
        print(f"{(p or 'all') + f' ({len(sel)})':14s} spellF1 {m[0]:.3f}  macroF1 {m[1]:.3f}  "
              f"alive/dead {m[2]:.3f}  team {m[3]:.3f}")


if __name__ == "__main__":
    main()
