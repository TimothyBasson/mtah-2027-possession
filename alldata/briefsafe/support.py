"""Are the extra brief possessions a run adds (vs a base run) real? Checked against each match's
own recorded ball: during the new possession, does the predicted team touch the ball (nearest
player within 1.5 m of the seen ball) more than the other team?

    python alldata/briefsafe/support.py --base pw1e --tag pcr1e
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "learned"))
import score as S  # noqa: E402
from score import pm, decode  # noqa: E402
import data as D  # noqa: E402


def spells(lab):
    e = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    s, t = np.r_[0, e], np.r_[e, len(lab)]
    return [(a, b, lab[a]) for a, b in zip(s, t) if lab[a] in "HA"]


def one(args):
    name, fb, ft = args
    raw, fold, spans = S.truths(name)
    d = np.load(D.match_path(name))
    ball, pl, team = d["ball"][:, :2], d["players"][:, :, :2], d["team"]
    dist = np.linalg.norm(pl - ball[:, None], axis=-1)
    dist = np.where(np.isnan(dist), np.inf, dist)
    near = dist.argmin(1)
    ok = np.isfinite(dist.min(1)) & (dist.min(1) < 1.5)
    touch = np.where(ok, team[near], -1)          # 0 home, 1 away, -1 nobody / ball unseen
    lab = {}
    for k, f in (("b", fb), ("t", ft)):
        z = np.load(f)
        lab[k] = decode.labels_from([z[f"s{i}"] for i in range(len(spans))], spans, len(raw), **S.RAW)
    truth = np.asarray(d["labels"]).astype("<U1")
    rows = []
    for a, b, k in spells(lab["t"]):
        if (b - a) / 25 >= 5 or (lab["b"][a:b] == k).mean() > 0.5:
            continue                                  # only short possessions the base gave to the other team
        own, oth = (touch[a:b] == (0 if k == "H" else 1)).sum(), (touch[a:b] == (1 if k == "H" else 0)).sum()
        rows.append([(truth[a:b] == k).mean() > 0.5, own > oth, own + oth == 0])
    return name, np.array(rows, float).reshape(-1, 3)


ap = argparse.ArgumentParser(); ap.add_argument("--base", default="pw1e"); ap.add_argument("--tag", default="pcr1e")
a = ap.parse_args()
with ProcessPoolExecutor(3) as ex:
    for pv in ("idsse", "metrica", "skillcorner"):
        jobs = [(os.path.basename(f)[:-4], f, os.path.join(S.tag_dir(a.tag), pv, os.path.basename(f)))
                for f in sorted(glob.glob(os.path.join(S.tag_dir(a.base), pv, f"{pv}_*.npz")))]
        R = np.concatenate([r for _, r in ex.map(one, jobs)])
        tr, sup, none = R[:, 0] > 0, R[:, 1] > 0, R[:, 2] > 0
        print(f"{pv:12s} new brief possessions per match {len(R) / len(jobs):5.1f}  label agrees {tr.mean():.2f}  "
              f"ball backs predicted team {sup.mean():.2f}  no touch seen {none.mean():.2f}  "
              f"label agrees | ball backs {tr[sup].mean():.2f}  ball backs | label agrees {sup[tr].mean():.2f}")
