"""Many variants of deep/, hmm v7 and their combination on the 30 outside matches, in one pass.

    python combined/iterate.py [--jobs 4] [--out logs/iterate.npz]

A variant is (source of deep/ probabilities, how it is used, clean-up of short runs):
  source   one or more tags under data/cache/deep (or a path), probabilities averaged
  use      "deep": deep/'s own decode at a switch cost; "hmm": hmm v7 with deep/ added as a
           clue at (weight where the ball is usable, weight where it is not); (0, 0) is hmm v7
  clean-up (shortest possession, shortest stoppage) in seconds, absorbed as deep/decode.absorb does
Parked on 2026-11-14: the combination with hmm waits until the players-only model is finished.
Every variant uses the same setting on every match; nothing is picked here. Scored against the
labels as they are and folded towards the controlled-possession rule. Saves every score per match.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "deep"), os.path.join(ROOT, "external"), os.path.join(ROOT, "evaluation")]
import decode  # noqa: E402
import score_control as sc  # noqa: E402
import score_hmm as sh  # noqa: E402
from control import control_labels  # noqa: E402
from hmm.clues import ball_is_usable, ball_place_category, gap_distance_category  # noqa: E402

dec = sc.dec
V2S1 = "/mnt/project-files/deep/v2s1"
SOURCES = {"v2": ["v2"], "v4": ["v4"], "v2+v4": ["v2", "v4"], "v2+s1": ["v2", V2S1], "v2+s1+v4": ["v2", V2S1, "v4"]}
CLEAN = ((1, 1), (2, 2), (2, 3), (3, 3))
VARIANTS = [(s, "deep", 4, c) for s in SOURCES for c in CLEAN]
VARIANTS += [("v2", "hmm", (0.0, 0.0), c) for c in ((0, 0),) + CLEAN]
VARIANTS += [(s, "hmm", w, c) for s in ("v2", "v2+v4", "v2+s1+v4") for w in ((0.0, 0.2), (0.05, 0.2), (0.05, 0.4), (0.1, 0.4))
             for c in ((0, 0), (2, 2), (2, 3))]


def name_of(v):
    s, use, w, c = v
    head = f"deep {s}" if use == "deep" else ("hmm v7" if w == (0.0, 0.0) else f"hmm + {s} at {w[0]:g}/{w[1]:g}")
    return f"{head}, min {c[0]:g}/{c[1]:g} s"


def source_probs(name, tags, spans):
    out = []
    for t in tags:
        base = t if os.path.isabs(t) else os.path.join(ROOT, "data", "cache", "deep", t)
        p = np.load(os.path.join(base, name.split("_")[0], name + ".npz"))
        out.append([p[f"p{i}"] for i in range(len(spans))])
    return [np.mean([o[i] for o in out], 0) for i in range(len(spans))]


def clean(pred, c, fps, spans):
    if not any(c):
        return pred.copy()
    codes = {"H": 0, "A": 1, "D": 2}
    out = pred.copy()
    for a, b in spans:
        seg = np.array([codes[x] for x in pred[a:b + 1]], np.int8)
        out[a:b + 1] = decode.LETTERS[decode.absorb(seg, {0: c[0] * fps, 1: c[0] * fps, 2: c[1] * fps})]
    return out


def one(name):
    paths, tabs, loso = (sc.STATE[k] for k in ("paths", "tables", "loso"))
    speed_t, flight_t, gap_t, width_t = tabs[name.split("_")[0]]
    d = sh.load(paths[name])
    ball = sc.match_ball(name, d)
    views = (d["labels"], control_labels(d, sc.STATE["fold_seconds"]))
    fps, pitch = int(d["fps"]), tuple(float(v) for v in d["pitch"])
    spans = [tuple(int(x) for x in h) for h in d["halves"]]
    segments = [(a, b + 1) for a, b in spans]
    bx, by = ball[:, 0].astype(float), ball[:, 1].astype(float)
    usable = ball_is_usable(bx, by, pitch)
    d_home, d_away = sh.nearest_by_team(d["players"], ball, d["team"])
    place = ball_place_category(bx, by, pitch, dec.CARRY_SECONDS * fps)
    clues = [(gap_distance_category(usable, d_home, d_away), gap_t),
             (place, dec.place_table()), (sh.speed_cats(d), speed_t), (sh.flight_cats(d, ball), flight_t)]
    extra = [(sh.width_cats(d), width_t, dec.WIDTH_WEIGHT)]
    prior = np.log(loso["prior_" + name])  # see combined/deep_in_hmm.py on these shares
    ratio = np.log(np.clip(loso[name], 1e-6, 1)) - prior
    trans, start = dec.transitions(), np.log(dec.START)
    probs = {s: source_probs(name, tags, spans) for s, tags in SOURCES.items()}
    base = {}
    for s, use, w, c in VARIANTS:
        key = (s, use, w)
        if key in base:
            continue
        if use == "deep":
            base[key] = decode.labels_from(probs[s], spans, len(d["labels"]), cost=w, min_team=0, min_dead=0)
        else:
            term = np.zeros_like(ratio)
            if any(w):
                for i, (a, b) in enumerate(spans):
                    term[a:b + 1] = np.log(np.clip(decode.upsample(probs[s][i], b - a + 1), 1e-6, 1)) - prior
            base[key] = dec.decode(clues, trans, start, segments, kickoffs=(), extra=extra,
                                   direct=dec.CONTROL_WEIGHT * ratio + np.where(usable, w[0], w[1])[:, None] * term)
    if sc.STATE.get("pseudo"):  # labels for deep/adapt.py, which never sees the true ones
        for (s, use, w), pred in base.items():
            if use == "hmm":
                tag = "hmm_v7" if not any(w) else f"hmm_{s}_{w[0]:g}_{w[1]:g}"
                os.makedirs(os.path.join(sc.STATE["pseudo"], tag), exist_ok=True)
                np.save(os.path.join(sc.STATE["pseudo"], tag, name + ".npy"), pred)
    out = []
    for s, use, w, c in VARIANTS:
        pred = clean(base[(s, use, w)], c, fps, spans)
        pred[d["labels"] == "X"] = "X"
        out.append([sh.score(gt, pred)[0] for gt in views])
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "iterate.npz"))
    ap.add_argument("--pseudo", action="store_true",
                    help="only hmm v7 and hmm + v2 at 0.05/0.4, and save their labels per match "
                         "under data/cache/pseudo/ for deep/adapt.py")
    a = ap.parse_args()
    if a.pseudo:
        VARIANTS[:] = [("v2", "hmm", w, (0, 0)) for w in ((0.0, 0.0), (0.05, 0.4))]
    paths = {os.path.basename(p)[:-4]: p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
             if os.path.basename(p).startswith(sh.SETS)}
    state = dict(paths=paths, tables=sc.tables(paths), loso=dict(np.load(os.path.join(sc.CACHE, "control_loso.npz"))),
                 weights=[dec.CONTROL_WEIGHT], fold_seconds=sc.CONTROL_SECONDS,
                 pseudo=os.path.join(ROOT, "data", "cache", "pseudo") if a.pseudo else None)
    with Pool(a.jobs, initializer=sc._init, initargs=(state,)) as pool:
        rows = dict(pool.imap_unordered(one, list(paths)))
    names = sorted(rows)
    S = np.array([rows[n] for n in names])  # (matches, variants, view)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, scores=S, matches=names, variants=[name_of(v) for v in VARIANTS])
    sets = [np.array([n.startswith(k) for n in names]) for k in sh.SETS]
    print(f"{'variant':44s}  raw: IDSSE Metrica SkillC  mean | folded: IDSSE Metrica SkillC  mean")
    for j, v in enumerate(VARIANTS):
        line = f"{name_of(v):44s} "
        for view in (0, 1):
            m = [S[k, j, view].mean() for k in sets]
            line += "      " + "   ".join(f"{x:.3f}" for x in m) + f"  {np.mean(m):.3f} " + ("|" if view == 0 else "")
        print(line)


if __name__ == "__main__":
    main()
