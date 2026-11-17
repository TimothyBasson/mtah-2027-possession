"""Feature tables for the 30 outside matches (and BAR-ATH), cached under data/cache/alldata/.

    python alldata/physics/build.py [--mode barath] [--prior bt bts1] [--only idsse_J03WMX ...]

One file per match and ball mode: features for the match as it is and with the teams exchanged,
the labels as built and folded to the controlled-possession rule (external/control.py, read from
the ball as recorded), and what scoring needs.
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import feats as FE  # noqa: E402

CODE = {"H": 0, "A": 1, "D": 2, "X": -1}


def table_path(name, mode, tag="f1"):
    return os.path.join(C.CACHE, tag, mode, name + ".npz")


def build_one(args):
    name, mode, prior_tags, tag, seed, knock = args
    out = table_path(name, mode, tag)
    if os.path.exists(out):
        return name
    d = C.load_bar_ath() if name == "bar_ath" else C.load_match(name)
    prior = C.prior_probs(name, prior_tags) if prior_tags else None
    dm = d if name == "bar_ath" else dict(d, ball=C.shown_ball(d, mode, seed))  # d keeps the recorded ball
    if knock and name != "bar_ath":
        dm["players"] = C.shown_players(d, seed)
    shown = np.isfinite(dm["ball"][:, 0])
    F, frames, half, names = FE.match_features(dm, shown, prior)
    save = dict(F=F, frames=frames, half=half, names=np.array(names), halves=np.array(d["halves"]),
                shown=shown)
    if name != "bar_ath":
        ds, ps = C.swap_teams(dm, prior) if prior is not None else (C.swap_teams(dm, [np.zeros((1, 3))])[0], None)
        save["Fs"] = FE.match_features(ds, shown, ps)[0]
        sys.path.insert(0, os.path.join(C.ROOT, "external"))
        from control import control_labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            folded = control_labels(d)
        save["labels"] = d["labels"]
        save["folded"] = folded
        live = (d["labels"] == "H") | (d["labels"] == "A")
        print(f"{name}: shown {shown[d['labels'] != 'X'].mean():.2f}, of live {shown[live].mean():.2f}, "
              f"of dead {shown[d['labels'] == 'D'].mean():.2f}", flush=True)
        save["y"] = np.array([CODE[c] for c in d["labels"][frames]], np.int8)
        save["yf"] = np.array([CODE[c] for c in folded[frames]], np.int8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", **save)
    os.replace(out + ".tmp.npz", out)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="barath")
    ap.add_argument("--prior", nargs="*", default=["bt"])
    ap.add_argument("--tag", default="f1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knock", action="store_true", help="lose players as BAR-ATH's tracker does")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    todo = a.only or C.names()
    with ProcessPoolExecutor(a.workers) as ex:
        for nm in ex.map(build_one, [(n, a.mode, tuple(a.prior), a.tag, a.seed, a.knock) for n in todo]):
            print("built", nm, flush=True)


if __name__ == "__main__":
    main()
