"""Held-out predictions of a possession model on every outside match, saved for the diagnostics.

    python diagnostics/predict.py --model possession --out data/cache/diag/main

For each provider the model's fit() runs on the other two (as external/heldout.py does), then each
match of that provider is labelled. Saved per match: the labels, the team call on every frame
(the labels with the stoppage model switched off), and the labels under weights fitted on all 30
matches (in-sample), to size the gap between the two.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "external")]
from score import frames  # noqa: E402

NEVER_DEAD = {"w": [-50.0, 0, 0, 0], "mu": [0, 0, 0], "sd": [1, 1, 1]}


def provider(path):
    return os.path.basename(path).split("_")[0]


def load_match(path):
    d = np.load(path)
    return (provider(path), d["players"], d["labels"], [tuple(h) for h in d["halves"]], d["team"])


def fit_fold(args):
    model_dir, train = args
    sys.path.insert(0, model_dir)
    import inspect
    import model
    ms = [load_match(p) for p in train]
    if "team" not in inspect.getsource(model.fit):  # main's fit takes four fields
        ms = [m[:4] for m in ms]
    return model.fit(ms)


def predict_match(args):
    path, model_dir, fits, out = args
    sys.path.insert(0, model_dir)
    import features
    import model
    d = np.load(path)
    halves = [tuple(h) for h in d["halves"]]
    mask = np.zeros(len(d["labels"]), bool)
    for s, e in halves:
        mask[s:e + 1] = True
    features.PITCH = model.PITCH = tuple(d["pitch"])
    model.HALVES = halves
    model.graded_mask = lambda n: mask
    F = frames(d["players"], d["ball"], d["team"], features)
    save = {}
    for name, params in fits.items():
        e = model.explain(F, params)
        save["pred_" + name] = e["pred"]
        if name == "heldout":
            for k in ("p_home", "p_dead"):
                if k in e:
                    save[k] = e[k].astype(np.float32)
            save["team"] = model.predict(F, params | {"dead": "logit", "dead_fit": NEVER_DEAD})
    np.savez_compressed(os.path.join(out, os.path.basename(path)), **save)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "possession"))
    ap.add_argument("--out", required=True)
    ap.add_argument("params", nargs="*")
    a = ap.parse_args()
    extra = {k: (v if v.isalpha() else float(v)) for k, v in (p.split("=") for p in a.params)}
    os.makedirs(a.out, exist_ok=True)
    model_dir = os.path.abspath(a.model)
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    sets = sorted({provider(p) for p in paths})
    with ProcessPoolExecutor(4) as ex:
        folds = [[p for p in paths if provider(p) != s] for s in sets] + [paths]
        fits = list(ex.map(fit_fold, [(model_dir, f) for f in folds]))
        by_set, insample = dict(zip(sets, fits[:-1])), fits[-1]
        for s in sets:
            print(s, by_set[s], flush=True)
        print("all", insample, flush=True)
        jobs = [(p, model_dir, {"heldout": extra | by_set[provider(p)], "insample": extra | insample}, a.out)
                for p in paths]
        for p in ex.map(predict_match, jobs):
            pass
    print("done", a.out)


if __name__ == "__main__":
    main()
