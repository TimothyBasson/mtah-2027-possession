"""SoccerTrack v2 -> data/soccertrack/soccertrack_<match>.npz, in the arrays external/build.py writes.

    python external/build_soccertrack.py --raw /path/to/raw [--out data/soccertrack]

SoccerTrack v2 (Scott, Uchida, Kuroda, Kim, Fujii; arXiv 2508.01802; CC BY 4.0) is 10 full
university matches filmed by one panoramic camera, with the provider's (Bepro) own tracking and
ball status under `raw/<match>/`:

    <match>_tracker_box_metadata.xml   teams, players (id, team), pitch size, periods
    <match>_tracker_box_data.xml       per 25 fps frame: ballStatus and every player's position

It is a fourth tracking system beside TRACAB (IDSSE), Metrica and SkillCorner: positions come as
pitch fractions to two decimals, i.e. on a 1.05 m x 0.68 m grid, with 22 players in almost every
frame. On that grid a player stands still in 87% of frames and jumps a metre in the rest, which
no other provider's tracking does (their frame-to-frame jitter is 0.2 to 1 cm) and which the
learned model reads as wild speeds (deep/shift.py: snapping any provider to this grid takes
0.23 off its spellF1). So positions are smoothed with a 1 s moving average applied twice
(SMOOTH_SECONDS; 0 keeps the grid), which turns the steps back into steady movement.

Labels. `ballStatus` is HOME / AWAY / BALLOUT / NEUTRAL per frame. BALLOUT is D. NEUTRAL (a loose
or contested ball, about 7% of frames, in runs of about 2 s) is held by the team that had the ball
before it, as SkillCorner's unnamed frames are in build.py; a NEUTRAL run straight after a
stoppage goes to the team that has it next, and X if none does. Like the other three sources it
is the provider's own rule, not the challenge's controlled-possession rule.

Where to get it: https://atomscott.github.io/SoccerTrack-v2/ (Hugging Face, gated, or the
authors' Google Drive mirror). Only the two XML files per match are read.
"""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25
SMOOTH_SECONDS = 1.0
FRAME = re.compile(r'<frame matchTime="[^"]*" frameNumber="([^"]*)" eventPeriod="([^"]*)" ballStatus="([^"]*)"')
PLAYER = re.compile(r'<(player|ball) playerId="([^"]*)" loc="\[([^,\]]*), ([^\]]*)\]"')
STATUS = {"HOME": "H", "AWAY": "A", "BALLOUT": "D", "NEUTRAL": "N"}


def speed(xy, fps, lag=5):
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    v[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1) * fps / lag
    return v


def smooth(xy, seconds, fps=FPS):
    """Centred moving average over `seconds` along time, skipping missing frames."""
    k = max(1, int(round(seconds * fps)) | 1)
    ok = np.isfinite(xy[..., :1])
    c = np.cumsum(np.concatenate([np.zeros_like(xy[:1]), np.where(ok, xy, 0)]), 0, dtype=np.float64)
    m = np.cumsum(np.concatenate([np.zeros((1,) + ok.shape[1:]), ok.astype(np.float64)]), 0)
    n, h = len(xy), k // 2
    lo, hi = np.clip(np.arange(n) - h, 0, n), np.clip(np.arange(n) + h + 1, 0, n)
    return np.where(ok, (c[hi] - c[lo]) / np.maximum(m[hi] - m[lo], 1), np.nan).astype(np.float32)


def hold_neutral(lab):
    """NEUTRAL runs -> the team before them; after a stoppage (or at the start) the team after."""
    out = lab.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != "N":
            i += 1
            continue
        j = i
        while j < n and out[j] == "N":
            j += 1
        before = out[i - 1] if i else "D"
        after = lab[j] if j < n else "D"
        out[i:j] = before if before in "HA" else (after if after in "HA" else "X")
        i = j
    return out


def build(meta_path, data_path):
    meta = ET.parse(meta_path).getroot()
    pitch = meta.find("pitch")
    L, W = float(pitch.get("width")), float(pitch.get("height"))
    side = {t.get("id"): 0 if t.get("side") == "home" else 1 for t in meta.iter("team")}
    team_of = {p.get("id"): side[p.get("teamId")] for p in meta.iter("player")}
    periods, frames, status = [], [], []
    rows = []  # (frame index, player id, x, y)
    ballrows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            m = FRAME.search(line)
            if m:
                periods.append(m.group(2))
                status.append(STATUS.get(m.group(3), "X"))
                frames.append(m.group(1))
                continue
            m = PLAYER.search(line)
            if m and periods:
                try:
                    x, y = float(m.group(3)), float(m.group(4))
                except ValueError:
                    continue
                (ballrows if m.group(1) == "ball" else rows).append((len(periods) - 1, m.group(2), x, y))
    n = len(periods)
    ids = sorted({r[1] for r in rows}, key=lambda p: (team_of.get(p, 9), p))
    ids = [p for p in ids if p in team_of]
    col = {p: k for k, p in enumerate(ids)}
    players = np.full((n, len(ids), 3), np.nan, np.float32)
    for i, p, x, y in rows:
        if p in col:
            players[i, col[p], 0], players[i, col[p], 1] = x * L, y * W
    ball = np.full((n, 3), np.nan, np.float32)
    for i, _, x, y in ballrows:
        ball[i, 0], ball[i, 1] = x * L, y * W
    periods = np.array(periods)
    labels = np.array(status, "<U1")
    halves = []
    for name in dict.fromkeys(periods.tolist()):  # in file order
        idx = np.flatnonzero(periods == name)
        a, b = int(idx[0]), int(idx[-1])
        assert b - a + 1 == len(idx), "a period's frames are not contiguous"
        labels[a:b + 1] = hold_neutral(labels[a:b + 1])
        if SMOOTH_SECONDS:
            for _ in range(2):
                players[a:b + 1, :, :2] = smooth(players[a:b + 1, :, :2], SMOOTH_SECONDS)
                ball[a:b + 1, :2] = smooth(ball[a:b + 1, :2], SMOOTH_SECONDS / 2)
        players[a:b + 1, :, 2] = speed(players[a:b + 1, :, :2], FPS)
        ball[a:b + 1, 2] = speed(ball[a:b + 1, :2], FPS)
        halves.append((a, b))
    team = np.array([team_of[p] for p in ids], np.int8)
    return players, ball, team, labels, np.array(halves), np.array([L, W]), FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder holding <match>_tracker_box_{metadata,data}.xml")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "soccertrack"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for meta in sorted(glob.glob(os.path.join(a.raw, "**", "*_tracker_box_metadata.xml"), recursive=True)):
        match = os.path.basename(meta).split("_")[0]
        dst = os.path.join(a.out, f"soccertrack_{match}.npz")
        data = meta.replace("_metadata.xml", "_data.xml")
        if os.path.exists(dst) or not os.path.exists(data):
            continue
        players, ball, team, labels, halves, pitch, fps = build(meta, data)
        if not len(labels) or not np.isin(labels, ["H", "A"]).any():
            print(f"soccertrack_{match}: no ball status in the tracking, skipped", flush=True)
            continue
        np.savez_compressed(dst + ".tmp.npz", players=players, ball=ball, team=team, labels=labels,
                            halves=halves, pitch=pitch, fps=fps)
        os.replace(dst + ".tmp.npz", dst)
        g = labels[labels != "X"]
        runs = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        ha = np.isin(labels[runs], ["H", "A"])
        print(f"soccertrack_{match}: {len(labels)} frames, {len(halves)} periods, {len(team)} players, "
              f"tracked per frame {np.isfinite(players[..., 0]).sum(1).mean():.1f}, ball seen "
              f"{np.isfinite(ball[:, 0]).mean():.0%}, labels " +
              " ".join(f"{k} {np.mean(g == k):.0%}" for k in "HAD") +
              f", {ha.sum()} possessions, median {np.median(np.diff(np.r_[runs, len(labels)])[ha]) / fps:.1f} s",
              flush=True)


if __name__ == "__main__":
    main()
