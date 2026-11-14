"""Lose players the way BAR-ATH's tracker loses them.

    python adapt/dropout.py --fit         # measure BAR-ATH, calibrate, write adapt/bar_ath_dropout.json
    python adapt/dropout.py --check       # the simulated outside matches beside BAR-ATH

What BAR-ATH's tracking shows, measured on the tracking alone (no labels):

- 7.6% of player-frames are missing. A track ends 1,198 times in the match and the player is
  then gone for seconds, not frames: almost no gap is under 1 s.
- Who is lost is far from random. A player standing (under 0.5 m/s) within 0.75 m of another is
  lost 13 times a minute; a player running (over 3 m/s) with nobody within 6 m, 0.007 times a
  minute. It is occlusion: 32% of losses happen within 1 m of another player, against 4% of all
  player-frames, at a median speed of 0.5 m/s against 1.4.
- The frames with 14 or fewer players tracked (5.3%) are not random bursts either. They are 33
  stretches bunched in a few stoppages (the goals, the end of a half): the ball is never seen in
  them and the players are walking. Players huddle, hide each other and are lost together.
- A player comes back when he moves again: tracks start mostly near another player but at
  ordinary speeds.

external/score.py drop_players loses each player at a flat rate whatever he is doing, and drops
its bursts anywhere, including on open play. simulate() below instead walks the match at 5 fps
and loses each visible player with BAR-ATH's measured rate for his situation (distance to the
nearest *visible* player, speed), keeps him lost for at least MIN_GAP seconds, then brings him
back at one rate while he stands or walks and a much higher one once he moves. The loss rates are BAR-ATH's as measured; the two
return rates cannot be measured directly (a lost player's situation is not seen), so they are
calibrated so the fully tracked outside matches (IDSSE, Metrica) come out with BAR-ATH's share of
missing player-frames and of frames with 14 or fewer tracked.

Not copied: BAR-ATH also shows more than 22 tracks on 2.4% of frames (extra people on the pitch
in stoppages), and its 46 slots swap players; the model is blind to slot order.
"""

import argparse
import glob
import json
import os
import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLE = os.path.join(HERE, "bar_ath_dropout.json")
STEP = 5                              # simulate every 5th frame
DIST = [0.75, 1.5, 3.0, 6.0]          # metres to the nearest visible player: bin edges
SPEED = [0.5, 1.5, 3.0]               # m/s: bin edges
MIN_GAP = 1.0                         # seconds a lost player stays lost at least


def nearest_visible(xy, vis):
    """(T, K) distance from each player to the nearest other visible player; 99 if none."""
    x = np.where(vis, xy[..., 0], np.nan)
    y = np.where(vis, xy[..., 1], np.nan)
    d = np.hypot(xy[..., 0][:, :, None] - x[:, None, :], xy[..., 1][:, :, None] - y[:, None, :])
    K = xy.shape[1]
    d[:, np.arange(K), np.arange(K)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmin(d, 2), nan=99.0)


def measure(players, halves, fps=25):
    """Loss counts and exposure by situation, and the per-frame tracked counts, from tracking alone."""
    ok = np.isfinite(players[..., 0])
    expo = np.zeros((len(DIST) + 1, len(SPEED) + 1))
    loss = np.zeros_like(expo)
    for a, b in halves:
        for s in range(a, b, 5000):
            f = np.arange(s, min(s + 5000, b))
            nn = nearest_visible(players[f, :, :2].astype(float), ok[f])
            di = np.digitize(nn, DIST)
            si = np.digitize(np.nan_to_num(players[f, :, 2]), SPEED)
            o, lost = ok[f], ok[f] & ~ok[f + 1]
            np.add.at(expo, (di[o], si[o]), 1)
            np.add.at(loss, (di[lost], si[lost]), 1)
    return loss, expo / fps  # exposure in player-seconds


def shape(players, team, halves, mask=None):
    """The numbers the simulation is held to: share of player-frames missing (of 11 a side), and
    of frames with <= 14, < 20 and >= 22 players tracked."""
    ok = np.isfinite(players[..., 0])
    g = np.zeros(len(ok), bool)
    for a, b in halves:
        g[a:b + 1] = True
    if mask is not None:
        g &= mask
    n = ok[g].sum(1)
    short = sum(np.clip(11 - ok[g][:, team == t].sum(1), 0, None).sum() for t in (0, 1))
    return dict(missing=short / (22 * g.sum()), le14=(n <= 14).mean(), lt20=(n < 20).mean(),
                full=(n >= 22).mean())


def load_table():
    with open(TABLE) as f:
        return json.load(f)


def simulate(players, halves, fps, name, table=None, seed=0):
    """A copy of players with NaN where the simulated tracker has lost them, speeds recomputed.
    Players already missing stay missing and take no part. Seeded by match name."""
    sys.path.insert(0, os.path.join(ROOT, "external"))
    from build import speed
    t = table or load_table()
    haz = np.array(t["loss_per_second"]) * STEP / fps          # per step
    back = np.array([t["return_per_second_slow"], t["return_per_second_moving"]]) * STEP / fps
    min_steps = int(round(MIN_GAP * fps / STEP))
    slow_below = t["slow_below"]
    rng = np.random.default_rng(zlib.crc32(f"{name} {seed}".encode()))
    out = players.copy()
    K = players.shape[1]
    for a, b in halves:
        idx = np.arange(a, b + 1, STEP)
        xy = players[idx, :, :2].astype(float)
        sp = np.nan_to_num(players[idx, :, 2].astype(float))
        there = np.isfinite(xy[..., 0])
        lost = np.zeros(K, bool)
        since = np.zeros(K, int)
        hide = np.zeros((len(idx), K), bool)
        si_all = np.digitize(sp, SPEED)
        for i in range(len(idx)):
            vis = there[i] & ~lost
            nn = nearest_visible(xy[i:i + 1], vis[None])[0]
            u = rng.random(K)
            go = vis & (u < haz[np.digitize(nn, DIST), si_all[i]])
            ret = lost & (since >= min_steps) & (u < back[(sp[i] >= slow_below).astype(int)])
            lost = (lost | go) & ~ret
            since = np.where(lost, since + 1, 0)
            hide[i] = lost
        full = np.repeat(hide, STEP, 0)[:b - a + 1]
        out[a:b + 1][full] = np.nan
    out[..., 2] = speed(out[..., :2], fps)
    for a, _ in halves:
        out[a:a + 5, :, 2] = np.nan
    return out


def bar_ath():
    sys.path[:0] = [os.path.join(ROOT, "possession"), ROOT]
    import load
    return load.load()


def fully_tracked():
    """The outside matches whose tracking has every player throughout: the calibration set."""
    return [p for p in sorted(glob.glob(os.path.join(ROOT, "data", "external", "*.npz")))
            if os.path.basename(p).split("_")[0] in ("idsse", "metrica")]


def pooled_shape(table, paths, seed=0):
    rows = []
    for p in paths:
        d = np.load(p)
        halves = [tuple(int(v) for v in h) for h in d["halves"]]
        k = simulate(d["players"], halves, int(d["fps"]), os.path.basename(p), table, seed)
        rows.append(shape(k, d["team"], halves))
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}


def fit():
    players, _, team, halves = bar_ath()
    loss, expo = measure(players, halves)
    target = shape(players, team, halves)
    target = {k: float(v) for k, v in target.items()}
    table = dict(dist_edges=DIST, speed_edges=SPEED, loss_per_second=(loss / expo).tolist(),
                 losses=loss.tolist(), bar_ath=target)
    paths = fully_tracked()
    best = None
    grid = [(thr, slow, moving) for thr in (1.0, 1.25) for slow in (0.002, 0.005) for moving in (0.6, 1.0)]
    with ProcessPoolExecutor(4) as ex:
        tables = [dict(table, slow_below=thr, return_per_second_slow=slow, return_per_second_moving=moving)
                  for thr, slow, moving in grid]
        for t, s in zip(tables, ex.map(pooled_shape, tables, [paths] * len(tables))):
            err = sum(abs(np.log(max(s[k], 1e-4) / target[k])) for k in target)
            print(f"standing below {t['slow_below']} m/s returns {t['return_per_second_slow']}/s, moving "
                  f"{t['return_per_second_moving']}/s: " + " ".join(f"{k} {v:.3f}" for k, v in s.items()), flush=True)
            if best is None or err < best[0]:
                best = (err, dict(t, simulated=s))
    with open(TABLE, "w") as f:
        json.dump(best[1], f, indent=1)
    print("BAR-ATH", target, "\nkept", {k: best[1][k] for k in ("slow_below", "return_per_second_slow", "return_per_second_moving")})


def check():
    t = load_table()
    print("BAR-ATH     ", " ".join(f"{k} {v:.3f}" for k, v in t["bar_ath"].items()))
    for prov in ("idsse", "metrica", "skillcorner"):
        paths = sorted(glob.glob(os.path.join(ROOT, "data", "external", prov + "_*.npz")))
        rows = []
        for p in paths:
            d = np.load(p)
            halves = [tuple(int(v) for v in h) for h in d["halves"]]
            seen = np.isfinite(d["players"][..., 0]).sum(1) >= 20  # SkillCorner: frames it has at all
            rows.append(shape(simulate(d["players"], halves, int(d["fps"]), os.path.basename(p), t),
                              d["team"], halves, seen))
        print(f"{prov:12s}", " ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in rows[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit()
    if a.check:
        check()
