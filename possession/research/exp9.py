"""Do the signals survive BAR-ATH-like missing players? Weights fitted on complete tracking
(and, second, on dropped tracking), scored on matches with players knocked out in runs."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats, feats2, lab
from lab import *
from concurrent.futures import ProcessPoolExecutor

DROP = os.path.join(ROOT, "data", "cache", "research_drop")

def knock(players, seed):
    """Each slot goes missing in runs (mean 5 s off, 60 s on); every ~60 s half the players vanish for ~4 s."""
    rng = np.random.default_rng(seed)
    n, K, _ = players.shape
    out = players.copy()
    for k in range(K):
        i, on = 0, True
        while i < n:
            L = 1 + int(rng.exponential(60 * 25 if on else 5 * 25))
            if not on: out[i:i + L, k] = np.nan
            i, on = i + L, not on
    i = int(rng.exponential(60 * 25))
    while i < n:
        L = 1 + int(rng.exponential(4 * 25))
        out[i:i + L, rng.random(K) < 0.5] = np.nan
        i += L + int(rng.exponential(60 * 25))
    return out

def one(path):
    os.makedirs(DROP, exist_ok=True)
    out = os.path.join(DROP, os.path.basename(path))
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True); return z["who"].item(), z["dead"].item(), float(z["miss"])
    d = np.load(path)
    P = knock(d["players"], abs(hash(os.path.basename(path))) % 2**31)
    g = d["labels"] != "X"
    miss = 1 - np.isfinite(P[g][:, :, 0]).sum() / np.isfinite(d["players"][g][:, :, 0]).sum()
    halves = [tuple(h) for h in d["halves"]]
    r = feats.compute(P, d["team"], halves, d["pitch"])
    r["who"].update(feats2.compute(P, d["team"], halves, d["pitch"]))
    np.savez_compressed(out, who=np.array(r["who"], dtype=object), dead=np.array(r["dead"], dtype=object), miss=miss)
    return r["who"], r["dead"], miss

if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
    ms = load()
    for m, p in zip(ms, paths): m["who"].update(feats2.cached(p))
    with ProcessPoolExecutor(4) as ex:
        dropped = list(ex.map(one, paths))
    print("player-frames knocked out:", round(float(np.mean([d[2] for d in dropped])), 3))
    md = [dict(m, who=d[0], dead=d[1]) for m, d in zip(ms, dropped)]
    V = ["s1", "s3", "d2"]; BD = (["sp_med", "acc_p80"], ["s0.5", "d2", "s3"])
    base = ["wid", "len", "vcx"]
    def mats(mm, who, dblocks):
        Xw = [expand(m, "who", who, V)[0] for m in mm]
        Xd = [np.column_stack([expand(m, "dead", n, v)[0] for n, v in dblocks]) for m in mm]
        return Xw, Xd
    def go(tag, who, dblocks):
        Xw, Xd = mats(ms, who, dblocks); Dw, Dd = mats(md, who, dblocks)
        res = {"full->full": [], "full->dropped": [], "dropped->dropped": []}
        for test in PROVIDERS:
            fw, fd = fit_fold(ms, Xw, test, "who", True), fit_fold(ms, Xd, test, "dead", False)
            gw, gd = fit_fold(md, Dw, test, "who", True), fit_fold(md, Dd, test, "dead", False)
            idx = [i for i, m in enumerate(ms) if m["prov"] == test]
            res["full->full"].append(np.mean([score(ms[i], decode(ms[i], apply(fw, Xw[i]), apply(fd, Xd[i]))) for i in idx], 0))
            res["full->dropped"].append(np.mean([score(ms[i], decode(ms[i], apply(fw, Dw[i]), apply(fd, Dd[i]))) for i in idx], 0))
            res["dropped->dropped"].append(np.mean([score(ms[i], decode(ms[i], apply(gw, Dw[i]), apply(gd, Dd[i]))) for i in idx], 0))
        print(f"{tag:34s} " + "  ".join(f"{k} {np.mean(v,0)[0]:.3f} (team {np.mean(v,0)[3]:.3f}, a/d {np.mean(v,0)[2]:.3f})" for k, v in res.items()), flush=True)
    go("spread only + 3 dead", ["spread"], [BD])
    go("3 who + 3 dead", base, [BD])
    go("who6 + 3 dead", base + ["speed", "press_max", "pol"], [BD])
    go("who6 - press_max", base + ["speed", "pol"], [BD])
    go("3 who + dead5", base, [BD, (["corner_min", "box_max", "spread_sum", "own_half", "sp_p80"], V)])
    go("who6 + dead5", base + ["speed", "press_max", "pol"], [BD, (["corner_min", "box_max", "spread_sum", "own_half", "sp_p80"], V)])
    go("who all + dead all", list(ms[0]["who"]), [BD, ([n for n in ms[0]["dead"] if n not in ("sp_med", "acc_p80")], V)])
