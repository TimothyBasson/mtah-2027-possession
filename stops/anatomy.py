"""Where the model goes wrong at each true stoppage: edge timing, missed and invented stoppages,
and the team named just after a restart, by restart kind (read off the ball: analysis only).

    python stops/anatomy.py bt
"""
import sys
from multiprocessing import Pool
import numpy as np
import common as C
from common import decode
from oracle import runs


def restart_kind(d, f, pitch):
    """Kind of restart at first live frame f, from where the ball is (seen within 1 s)."""
    L, W = pitch
    b = d["ball"][f:f + 25, :2]
    ok = np.isfinite(b[:, 0])
    if not ok.any():
        return "unseen"
    x, y = b[ok][0]
    x, y = abs(x - L / 2), abs(y - W / 2)
    if x > L / 2 - 3 and y > W / 2 - 3:
        return "corner"
    if y > W / 2 - 1.5:
        return "throw-in"
    if x > L / 2 - 8 and y < 12:
        return "goal kick"
    if x < 2 and y < 2:
        return "kick-off"
    return "free kick"


def one(args):
    name, tag, T = args
    d = np.load(f"{C.ROOT}/data/external/{name}.npz")
    truth = T["raw"]
    pred = decode.labels_from(C.probs(tag, name), T["spans"], len(truth), **C.DECODE)
    rows = []   # per true stoppage: len, start offset, end offset (pred - true, frames; nan if missed), kind, team ok first 2 s, team before ok
    for a0, b0 in T["spans"]:
        for a, b, k in runs(truth[a0:b0 + 1]):
            a, b = a + a0, b + a0
            if k != "D" or a == a0 or b > b0 or truth[a - 1] not in "HA" or truth[b] not in "HA":
                continue
            pd = pred[a:b] == "D"
            so = eo = np.nan
            if pd.mean() > 0.3:
                i = a + np.flatnonzero(pd)[0]
                while i > a0 and pred[i - 1] == "D":
                    i -= 1
                j = a + np.flatnonzero(pd)[-1]
                while j < b0 and pred[j + 1] == "D":
                    j += 1
                so, eo = i - a, j + 1 - b
            e = b
            while e <= b0 and truth[e] == truth[b]:
                e += 1
            seg = slice(b, min(e, b + 50))
            after = (pred[seg] == truth[b]).mean()
            s = a - 1
            while s > a0 and truth[s - 1] == truth[a - 1]:
                s -= 1
            before = (pred[max(s, a - 50):a] == truth[a - 1]).mean()
            rows.append((b - a, so, eo, restart_kind(d, b, tuple(d["pitch"])), after, before,
                         truth[a - 1] == truth[b], e - b, a - s))
    # invented stoppages: predicted D runs with < 30% true D
    inv = [(b - a) for a, b, k in runs(pred) if k == "D" and (truth[a:b] == "D").mean() < 0.3 and (truth[a:b] != "X").all()]
    return name, rows, inv


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "bt"
    T = C.truths()
    with Pool(4) as pool:
        out = pool.map(one, [(n, tag, T[n]) for n in sorted(T)])
    for p in C.PROVIDERS:
        rows = [r for n, rs, _ in out if n.startswith(p) for r in rs]
        inv = [x for n, _, i in out if n.startswith(p) for x in i]
        nm = sum(n.startswith(p) for n, _, _ in out)
        ln = np.array([r[0] for r in rows]) / 25
        so = np.array([r[1] for r in rows]) / 25
        eo = np.array([r[2] for r in rows]) / 25
        found = np.isfinite(so)
        print(f"\n{p}: {len(rows) / nm:.0f} true stoppages a match, median {np.median(ln):.1f} s; invented {len(inv) / nm:.0f} a match (median {np.median(inv) / 25:.1f} s)")
        for lo, hi in ((0, 2), (2, 5), (5, 15), (15, 1e9)):
            m = (ln >= lo) & (ln < hi)
            print(f"  true length {lo:>2}-{hi if hi < 1e8 else 'inf':>3} s: {m.sum() / nm:5.1f} a match, found {found[m].mean():.0%}")
        q = lambda v: "  ".join(f"{x:+.1f}" for x in np.nanpercentile(v, [10, 25, 50, 75, 90]))
        print(f"  start offset s (pred - true) p10..p90: {q(so)}   |off|<=0.5 s: {(np.abs(so[found]) <= .5).mean():.0%}")
        print(f"  end   offset s (pred - true) p10..p90: {q(eo)}   |off|<=0.5 s: {(np.abs(eo[found]) <= .5).mean():.0%}")
        same = np.array([r[6] for r in rows])
        print(f"  same team before and after: {same.mean():.0%}")
        print(f"  team right in last 2 s before stop: {np.mean([r[5] for r in rows]):.0%}   (possession before under 2 s: {np.mean([r[8] < 50 for r in rows]):.0%} of stops, right {np.mean([r[5] for r in rows if r[8] < 50]):.0%})")
        for k in ("throw-in", "corner", "goal kick", "kick-off", "free kick", "unseen"):
            rr = [r for r in rows if r[3] == k]
            if rr:
                f_ = np.isfinite([r[1] for r in rr])
                print(f"  {k:10s} {len(rr) / nm:5.1f} a match  found {f_.mean():.0%}  team right in first 2 s: {np.mean([r[4] for r in rr]):.0%}  "
                      f"median len {np.median([r[0] for r in rr]) / 25:.1f} s  end off median {np.nanmedian([r[2] for r in rr]) / 25:+.1f}  first poss <2 s: {np.mean([r[7] < 50 for r in rr]):.0%}")
