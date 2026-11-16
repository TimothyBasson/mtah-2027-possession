import numpy as np, sys
from multiprocessing import Pool
import common as C
from common import sh
from hmm.ball import detect_touches
TAGS = ["bt", "bts1"]
def one(path):
    name = path.split("/")[-1][:-4]
    d = sh.load(path)
    spans = [tuple(h) for h in d["halves"]]
    n = len(d["labels"])
    pred = C.labels_ev(C.base_probs(name, TAGS, len(spans)), spans, n)
    gt = d["labels"]
    tv = C.turnovers(gt); pf = C.turnovers(pred)
    off = []
    for f in pf:
        c = tv[gt[tv] == pred[f]]
        if len(c):
            k = c[np.abs(c - f).argmin()] ; off.append(f - k)
    # true turnover vs nearest ball touch by the winning team
    ball = d["ball"].astype(float); pl = d["players"].astype(float)
    dist = np.hypot(pl[:, :, 0] - ball[:, None, 0], pl[:, :, 1] - ball[:, None, 1])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dh = np.nanmin(dist[:, d["team"] == 0], 1); da = np.nanmin(dist[:, d["team"] == 1], 1)
    t = detect_touches(ball[:, 0], ball[:, 1], dh, da, 25)
    toff = []
    for f in tv:
        c = t["frame"][t["team"] == (0 if gt[f] == "H" else 1)]
        if len(c):
            toff.append(c[np.abs(c - f).argmin()] - f)
    # what the model says at true turnovers: pred state before/after
    seen = np.isfinite(ball[tv, 0]).mean()
    return name, np.array(off), np.array(toff), seen
with Pool(4) as p: rows = p.map(one, C.paths())
edges = np.array([-75,-50,-37,-25,-12,-6,7,13,26,38,51,76])
for pr in C.PROVIDERS:
    o = np.concatenate([r[1] for r in rows if r[0].startswith(pr)]); t = np.concatenate([r[2] for r in rows if r[0].startswith(pr)])
    print(pr, "pred - nearest true (frames):", np.round(np.histogram(o, edges)[0] / len(o), 2), "median", np.median(o[np.abs(o) < 75]))
    print("   winner's nearest touch - true turnover:", np.round(np.histogram(t, edges)[0] / len(t), 2), "median", np.median(t[np.abs(t) < 75]), "ball seen at turnover", np.mean([r[3] for r in rows if r[0].startswith(pr)]).round(2))
