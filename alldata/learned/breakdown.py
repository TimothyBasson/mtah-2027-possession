"""Misses of a saved held-out run by situation: what kind of true possession is not found.

    python alldata/learned/breakdown.py --tag pw1e [--key s] [--max-len 1e9] [--out file.npz]

For every true possession (folded labels by default) of every held-out match: was it matched at
IoU 0.5 by the prediction (score.py's fixed decode), and what was the situation. The situation is
read from the recorded ball and players (not from what the model saw), except `seen`, which is
the share of the possession in which the BAR-ATH-like ball (build.ball_version 1) is visible.

  len      true length
  before   what precedes it: the other team (turnover) or a stoppage (restart)
  after    what follows it
  seen     share of its frames with a visible ball in the model's input
  third    where the ball is, along the possessing team's direction of attack
  lane     central or wide (outside the width of the penalty area)
  speed    the ball's mean speed over the second before it starts (the sources give no height)
  between  what it sits between: the same opponent possession resumed after it (a brief steal),
           two stoppages, or anything else
  prob     the net's mean probability of the true team inside it (before decoding)
  near     share of its frames in which the player nearest the recorded ball is of the true team
  nearsim  the same against the BAR-ATH-like ball the model saw (where it is visible)
  contact  share of its frames with a player of the true team within 2 m of the recorded ball
  travel   how far the ball moves inside it (path length)
  arrive   how far the ball travelled in the 2 s before it starts
  press    nearest opponent to the ball over its first second
  crowd    players within 5 m of the ball over its first second

Per value: possessions, share found, share of all misses, and the oracle gain: spellF1 if the
prediction were made right inside exactly those possessions (mean of the three providers).
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402
from score import pm, decode, D  # noqa: E402

H, A, Dd, X = (int(v) for v in pm.label_codes(np.array(list("HADX"))))
ATTRS = ["len", "before", "after", "seen", "third", "lane", "speed", "between", "prob", "near", "nearsim", "contact", "travel", "arrive", "press", "crowd", "how"]
VALUES = {
    "len": ["under 2 s", "2 to 5 s", "5 to 10 s", "10 to 20 s", "over 20 s"],
    "before": ["turnover", "restart", "start"],
    "after": ["turnover", "stoppage", "end"],
    "seen": ["hidden (<20%)", "partly", "seen (>80%)"],
    "third": ["own third", "middle third", "final third", "unknown"],
    "lane": ["central", "wide", "unknown"],
    "speed": ["under 5 m/s", "5 to 12 m/s", "over 12 m/s", "unknown"],
    "between": ["opponent both sides", "stoppage both sides", "other"],
    "near": ["under 40%", "40 to 70%", "over 70%", "unknown"],
    "nearsim": ["under 40%", "40 to 70%", "over 70%", "unknown"],
    "contact": ["under 20%", "20 to 60%", "over 60%", "unknown"],
    "travel": ["under 10 m", "10 to 30 m", "over 30 m", "unknown"],
    "prob": ["under 0.2", "0.2 to 0.4", "0.4 to 0.6", "over 0.6"],
    "arrive": ["under 8 m", "8 to 20 m", "over 20 m", "unknown"],
    "press": ["under 2 m", "2 to 5 m", "over 5 m", "unknown"],
    "crowd": ["0 to 2", "3 to 4", "5 or more", "unknown"],
    "how": ["found", "wrong team", "called dead", "merged into a longer one", "cut up or cut short", "edges off"],
}


def binned(x, edges):
    return int(np.searchsorted(edges, x, "right"))


def one(args):
    name, f, key, pre, max_len, only = args
    import importlib.util
    spec = importlib.util.spec_from_file_location("alldata_build", os.path.join(HERE, "build.py"))  # external/build.py shadows the name
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    raw, fold, spans = S.truths(name)
    t = fold if pre == "f" else raw
    dec = S.FOLDED if pre == "f" else S.RAW
    z = np.load(f)
    p = pm.label_codes(decode.labels_from([z[f"{pre}{key}{i}"] for i in range(len(spans))], spans, len(t), **dec))
    P = np.full((len(t), 3), 1 / 3)
    for i, (h0, h1) in enumerate(spans):
        P[h0:h1 + 1] = decode.upsample(z[f"{pre}{key}{i}"], h1 - h0 + 1)
    d = dict(np.load(D.match_path(name)))
    L, W = (float(v) for v in d["pitch"])
    fps = int(d["fps"])
    ball = d["ball"].astype(np.float64)
    sim = build.ball_version(d, name, 1).astype(np.float64)
    shown = np.isfinite(sim[:, 0])
    pl, team = d["players"][:, :, :2].astype(np.float64), d["team"]
    left = np.zeros(len(t), bool)  # home defends x = 0
    for a, b in spans:
        left[a:b + 1] = D.home_defends_left(d["players"], team, a, b)
    runs = pm.valid_runs(t)
    gts = pm.possession_spells(t, runs)
    prs = pm.possession_spells(p, runs)
    ov = pm.spell_overlaps(gts, prs)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gts), th) for th in pm.SPELL_IOUS])
    best = {}
    for iou, i, j in ov:
        best.setdefault(i, (iou, j))
    rows = []
    for i, (a, b, k) in enumerate(gts):
        r = {}
        r["len"] = binned((b - a) / fps, [2, 5, 10, 20])
        bef = t[a - 1] if a > 0 else X
        aft = t[b] if b < len(t) else X
        r["before"] = 0 if bef in (H, A) else 1 if bef == Dd else 2
        r["after"] = 0 if aft in (H, A) else 1 if aft == Dd else 2
        r["seen"] = binned(shown[a:b].mean(), [0.2, 0.8])
        bx, by, bz = ball[a:b].T
        if np.isfinite(bx).any():
            u = np.nanmean(bx) / L
            if (k == H) != bool(left[a]):
                u = 1 - u
            r["third"] = binned(u, [1 / 3, 2 / 3])
            r["lane"] = int(abs(np.nanmean(by) - W / 2) > 20.16)
        else:
            r["third"], r["lane"] = 3, 2
        zz = ball[max(a - fps, 0):a + 1, 2]
        r["speed"] = binned(np.nanmean(zz), [5, 12]) if np.isfinite(zz).any() else 3
        r["between"] = 0 if (bef in (H, A) and aft in (H, A)) else 1 if (bef == Dd and aft == Dd) else 2
        r["prob"] = binned(P[a:b, 0 if k == H else 1].mean(), [0.2, 0.4, 0.6])
        a0 = max(a - 2 * fps, 0)
        if np.isfinite(ball[a0, 0]) and np.isfinite(ball[a, 0]):
            r["arrive"] = binned(np.hypot(*(ball[a, :2] - ball[a0, :2])), [8, 20])
        else:
            r["arrive"] = 3
        own = team == (0 if k == H else 1)
        for nm, bb in (("near", ball), ("nearsim", sim)):
            dd = np.hypot(pl[a:b, :, 0] - bb[a:b, None, 0], pl[a:b, :, 1] - bb[a:b, None, 1])
            dd = np.where(np.isfinite(dd), dd, np.inf)
            ok = np.isfinite(bb[a:b, 0]) & np.isfinite(dd.min(1))
            do, dp = dd[:, own].min(1), dd[:, ~own].min(1)
            r[nm] = binned((do < dp)[ok].mean(), [0.4, 0.7]) if ok.any() else 3
            if nm == "near":
                r["contact"] = binned((do < 2)[ok].mean(), [0.2, 0.6]) if ok.any() else 3
        st = np.hypot(*np.diff(ball[a:b, :2], axis=0).T) if b - a > 1 else np.array([np.nan])
        r["travel"] = binned(np.nansum(st), [10, 30]) if np.isfinite(st).mean() > 0.8 else 3
        e = min(a + fps, b)
        dist = np.hypot(pl[a:e, :, 0] - ball[a:e, None, 0], pl[a:e, :, 1] - ball[a:e, None, 1])  # (T, K)
        opp = team == (1 if k == H else 0)
        if np.isfinite(dist).any() and np.isfinite(dist[:, opp]).any():
            r["press"] = binned(np.nanmean(np.nanmin(np.where(np.isfinite(dist[:, opp]), dist[:, opp], np.inf), 1)[np.isfinite(ball[a:e, 0])]), [2, 5])
            r["crowd"] = binned(np.nanmean((dist < 5).sum(1)[np.isfinite(ball[a:e, 0])]), [2.5, 4.5])
        else:
            r["press"], r["crowd"] = 3, 3
        seg = p[a:b]
        other = A if k == H else H
        if flags[pm.SPELL_IOUS.index(0.5) if 0.5 in pm.SPELL_IOUS else 0, i]:
            w = 0
        elif (seg == other).mean() > 0.5:
            w = 1
        elif (seg == Dd).mean() > 0.5:
            w = 2
        elif i in best and (prs[best[i][1]][1] - prs[best[i][1]][0]) > 2 * (b - a):
            w = 3
        elif (seg == k).mean() < 0.7:
            w = 4
        else:
            w = 5
        r["how"] = w
        rows.append([r[c] for c in ATTRS] + [int(w == 0) if only < 0 else int(w != only), (b - a) / fps, a, b, k])
    rows = np.array(rows, float)
    base = pm.evaluate(t, p, bootstrap_n=0)["spell_f1_mean"]
    gain = {}
    short = rows[:, len(ATTRS) + 1] < max_len
    for ci, c in enumerate(ATTRS[:-1]):
        for v in range(len(VALUES[c])):
            q = p.copy()
            for row in rows[(rows[:, ci] == v) & short & (rows[:, len(ATTRS)] == 0)]:
                q[int(row[-3]):int(row[-2])] = int(row[-1])
            gain[(c, v)] = pm.evaluate(t, q, bootstrap_n=0)["spell_f1_mean"] - base
    return name, rows, gain, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--key", default="s")
    ap.add_argument("--labels", default="folded", choices=["folded", "built"])
    ap.add_argument("--max-len", type=float, default=1e9, help="only possessions shorter than this many seconds")
    ap.add_argument("--only", type=int, default=-1, help="count as missed only the misses of this kind (index into the 'how' list: 1 wrong team, 2 called dead, ...)")
    ap.add_argument("--out")
    a = ap.parse_args()
    pre = "f" if a.labels == "folded" else ""
    jobs = []
    for fold in ("idsse", "metrica", "skillcorner"):
        for f in sorted(glob.glob(os.path.join(S.tag_dir(a.tag), fold, "*.npz"))):
            if str(np.load(f)["kind"]) == "test":
                jobs.append((os.path.basename(f)[:-4], f, a.key, pre, a.max_len, a.only))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    provs = ["idsse", "metrica", "skillcorner"]
    R = {pv: np.vstack([r for n, r, _, _ in res if n.startswith(pv)]) for pv in provs}
    if a.out:
        np.savez_compressed(a.out, attrs=np.array(ATTRS), **R)
    hi = len(ATTRS)
    print(f"{a.tag}, labels {a.labels}, possessions under {a.max_len:g} s; base " +
          " / ".join(f"{np.mean([b for n, _, _, b in res if n.startswith(pv)]):.3f}" for pv in provs))
    print("per provider: possessions, found at IoU 0.5, share of that provider's misses; then the oracle gain (mean of providers)")
    for ci, c in enumerate(ATTRS):
        print(f"\n-- {c}")
        for v, vn in enumerate(VALUES[c]):
            line = f"  {vn:26s}"
            for pv in provs:
                r = R[pv][R[pv][:, hi + 1] < a.max_len]
                m = r[:, ci] == v
                miss = (r[:, hi] == 0).sum()
                line += f" {int(m.sum()):5d} {r[m, hi].mean() if m.any() else float('nan'):5.2f} {((r[m, hi] == 0).sum() / max(miss, 1)):5.2f} |"
            if c != "how":
                g = np.mean([np.mean([gn[(c, v)] for n, _, gn, _ in res if n.startswith(pv)]) for pv in provs])
                line += f"  {g:+.3f}"
            print(line)


if __name__ == "__main__":
    main()
