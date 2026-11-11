"""Out-of-sample matches: other games' tracking, in the arrays possession/load.py gives for ours,
plus a per-frame H/A/D label from that provider's own data. RULES.md allows labels on other games.

    uv run python external/build.py [idsse] [metrica]     # -> data/external/<name>.npz

Each npz:
    players  (n, K, 3) float32  x, y in metres from a corner, speed m/s; NaN where missing
    ball     (n, 3)    float32  x, y in metres, speed; NaN where missing
    team     (K,)      int8     0 home, 1 away
    labels   (n,)      <U1      H / A / D, X outside the halves
    halves   (h, 2)    int      first and last frame of each period, inclusive
    pitch    (2,)      float    length, width in metres
    fps      ()        int

Labels:
    idsse    DFL's own BallStatus / BallPossession flag on every frame (data-operator definition).
    metrica  rebuilt from events: dead from a ball-out / foul / shot-out until the next restart,
             until the next on-ball action (pass, carry, shot, recovery, set piece), whose team holds
             it after. Challenges and BALL LOST don't move it.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from kloppy import metrica, sportec
from kloppy.domain import BallState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "external")
RAW = os.path.join(OUT, "raw")


def arrays(ds):
    """players, ball, team, halves, pitch from a kloppy TrackingDataset, frame i = records[i]."""
    m = ds.metadata
    pitch = np.array([m.pitch_dimensions.pitch_length, m.pitch_dimensions.pitch_width], float)
    ds = ds.transform(to_coordinate_system="kloppy")  # 0-1 from the top-left corner
    ids = [(p.player_id, k) for k, t in enumerate(m.teams) for p in t.players]
    col = {pid: i for i, (pid, _) in enumerate(ids)}
    n = len(ds.records)
    players = np.full((n, len(ids), 3), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    period = np.zeros(n, int)
    for i, f in enumerate(ds.records):
        period[i] = f.period.id
        if f.ball_coordinates is not None:
            ball[i, :2] = f.ball_coordinates.x, f.ball_coordinates.y
        for p, d in f.players_data.items():
            if d.coordinates is not None:
                players[i, col[p.player_id], :2] = d.coordinates.x, d.coordinates.y
    players[..., :2] *= pitch
    ball[:, :2] *= pitch
    fps = int(round(m.frame_rate))
    players[..., 2] = speed(players[..., :2], fps)
    ball[:, 2] = speed(ball[:, :2], fps)
    halves = np.array([(np.flatnonzero(period == k)[[0, -1]]) for k in np.unique(period)])
    team = np.array([k for _, k in ids], np.int8)
    return players, ball, team, halves, pitch, fps


def speed(xy, fps, lag=5):
    """m/s over `lag` frames, NaN where either end is missing."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    v[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1) * fps / lag
    return v


def save(name, players, ball, team, labels, halves, pitch, fps):
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), players=players, ball=ball, team=team,
                        labels=labels, halves=halves, pitch=pitch, fps=fps)
    g = labels[labels != "X"]
    print(f"{name}: {len(labels)} frames, {len(team)} players, ball seen "
          f"{np.isfinite(ball[:, 0]).mean():.0%}, labels " +
          " ".join(f"{k} {np.mean(g == k):.0%}" for k in "HAD"))


# -- IDSSE / DFL -------------------------------------------------------------------------------

def build_idsse():
    for meta in sorted(glob.glob(os.path.join(RAW, "idsse", "*matchinformation*.xml"))):
        mid = meta.rsplit("-", 1)[1][:-4]
        raw = glob.glob(os.path.join(RAW, "idsse", f"*positions*{mid}.xml"))[0]
        ds = sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False)
        players, ball, team, halves, pitch, fps = arrays(ds)
        home = ds.metadata.teams[0]
        labels = np.full(len(ds.records), "X", "<U1")
        for i, f in enumerate(ds.records):
            if f.ball_state == BallState.DEAD:
                labels[i] = "D"
            elif f.ball_owning_team is not None:
                labels[i] = "H" if f.ball_owning_team == home else "A"
        save(f"idsse_{mid}", players, ball, team, labels, halves, pitch, fps)


# -- Metrica sample games ----------------------------------------------------------------------

POSSESS = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


def metrica_labels(events, frame_index, halves):
    """events: (team 0/1, type, subtype, start_frame, end_frame), frames as in the tracking.
    Dead from a stoppage until the next on-ball action; while alive, that action's team."""
    marks = []  # (frame, order, what, side): order puts a touch after a stoppage on the same frame
    for side, kind, sub, s, e in events:
        s = frame_index.get(s)
        if s is None:
            continue
        e = frame_index.get(e, s)
        if kind in POSSESS:
            marks.append((s, 1, "touch", side))
        if kind == "BALL OUT" or (kind == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
            marks.append((e, 0, "stop", side))
        elif kind == "FAULT RECEIVED":
            marks.append((s, 0, "stop", side))
    marks.sort()
    state = np.full(halves[-1][1] + 1, "D", "<U1")  # before the first restart: not yet kicked off
    cur = "D"
    for (f, _, what, side), nxt in zip(marks, marks[1:] + [(len(state),)]):
        if what == "stop":
            cur = "D"
        else:  # the first on-ball action after a stoppage is the restart (goal kicks are logged as
            cur = "HA"[side]  # RECOVERY, not SET PIECE, so a restart can't be told apart)
        state[f:nxt[0]] = cur
    labels = np.full(len(state), "X", "<U1")
    for a, b in halves:
        labels[a:b + 1] = state[a:b + 1]
    return labels


def build_metrica():
    base = os.path.join(RAW, "metrica", "sample-data", "data")
    for g in (1, 2, 3):
        d = os.path.join(base, f"Sample_Game_{g}")
        if g < 3:
            ds = metrica.load_tracking_csv(
                home_data=os.path.join(d, f"Sample_Game_{g}_RawTrackingData_Home_Team.csv"),
                away_data=os.path.join(d, f"Sample_Game_{g}_RawTrackingData_Away_Team.csv"))
            ev = pd.read_csv(os.path.join(d, f"Sample_Game_{g}_RawEventsData.csv"))
            events = [(0 if r.Team == "Home" else 1, r.Type, r.Subtype if isinstance(r.Subtype, str) else "",
                       int(r["Start Frame"]), int(r["End Frame"])) for _, r in ev.iterrows()]
        else:
            ds = metrica.load_tracking_epts(meta_data=os.path.join(d, "Sample_Game_3_metadata.xml"),
                                            raw_data=os.path.join(d, "Sample_Game_3_tracking.txt"))
            raw = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]
            home = ds.metadata.teams[0].team_id
            events = [(0 if r["team"]["id"] == home else 1, r["type"]["name"], subtype(r),
                       r["start"]["frame"], r["end"]["frame"]) for r in raw]
        players, ball, team, halves, pitch, fps = arrays(ds)
        frame_index = {f.frame_id: i for i, f in enumerate(ds.records)}
        labels = metrica_labels(events, frame_index, halves)
        save(f"metrica_{g}", players, ball, team, labels, halves, pitch, fps)


def subtype(r):
    s = r.get("subtypes")
    if not s:
        return ""
    return "-".join(x["name"] for x in s) if isinstance(s, list) else s["name"]


# -- SkillCorner open data (A-League 2024-25, broadcast tracking, 10 fps) ---------------------

SC_FPS, FPS = 10, 25


def skillcorner_labels(group, alive, phase_team):
    """Per source frame. Dead outside a phase of play. Inside, the possession group, held through
    frames where SkillCorner names nobody (passes, loose balls); the phase's team until one is named.
    group, phase_team: 0 home, 1 away, -1 none."""
    labels = np.full(len(group), "D", "<U1")
    last = -1
    for i in range(len(group)):
        if not alive[i]:
            last = -1
            continue
        if group[i] >= 0:
            last = group[i]
        labels[i] = "HA"[last if last >= 0 else phase_team[i]]
    return labels


def resample(x, n):
    """10 fps -> 25 fps by linear interpolation along axis 0. NaN between two samples when either is
    NaN (no filling across gaps, like our ATD), but a sample that lands exactly is kept."""
    t = np.arange(n) * SC_FPS / FPS
    i = np.minimum(t.astype(int), len(x) - 2)
    w = (t - i).reshape((-1,) + (1,) * (x.ndim - 1)).astype(np.float32)
    return np.where(w == 0, x[i], x[i] * (1 - w) + x[i + 1] * w)  # NaN * 0 is NaN


def build_skillcorner():
    base = os.path.join(RAW, "skillcorner")
    for mid in sorted(json.load(open(os.path.join(base, "matches.json"))), key=lambda m: m["id"]):
        mid = str(mid["id"])
        d = os.path.join(base, mid)
        info = json.load(open(os.path.join(d, f"{mid}_match.json")))
        home = info["home_team"]["id"]
        pitch = np.array([info["pitch_length"], info["pitch_width"]], float)
        roster = sorted(info["players"], key=lambda p: (p["team_id"] != home, p["id"]))
        col = {p["id"]: k for k, p in enumerate(roster)}
        team = np.array([p["team_id"] != home for p in roster], np.int8)

        rows = [json.loads(line) for line in open(os.path.join(d, f"{mid}_tracking_extrapolated.jsonl"))]
        n = rows[-1]["frame"] + 1
        assert [r["frame"] for r in rows] == list(range(n)), f"{mid}: frames not contiguous"
        xy = np.full((n, len(roster), 2), np.nan, np.float32)
        ball = np.full((n, 2), np.nan, np.float32)
        period = np.zeros(n, int)
        group = np.full(n, -1, np.int8)
        for f, r in enumerate(rows):
            period[f] = r["period"] or 0
            group[f] = {"home team": 0, "away team": 1}.get(r["possession"]["group"], -1)
            b = r["ball_data"]
            if b.get("is_detected") and b["x"] is not None:  # extrapolated ball is a guess: our ATD has none
                ball[f] = b["x"], b["y"]
            for p in r["player_data"] or ():  # off-camera players are extrapolated; ours are ~always tracked
                if p["x"] is not None:
                    xy[f, col[p["player_id"]]] = p["x"], p["y"]
        xy += pitch / 2  # centre origin -> corner
        ball += pitch / 2

        ph = pd.read_csv(os.path.join(d, f"{mid}_phases_of_play.csv"))
        alive = np.zeros(n, bool)
        phase_team = np.zeros(n, np.int8)
        for s, e, t in zip(ph.frame_start, ph.frame_end, ph.team_in_possession_id):
            alive[s:e + 1] = True
            phase_team[s:e + 1] = t != home
        lab10 = skillcorner_labels(group, alive, phase_team)

        parts, halves, start = [], [], 0
        for k in (1, 2):
            sel = np.flatnonzero(period == k)
            sel = np.arange(sel[0], sel[-1] + 1)
            m = (len(sel) - 1) * FPS // SC_FPS + 1
            parts.append((resample(xy[sel], m), resample(ball[sel], m),
                          lab10[sel][np.arange(m) * SC_FPS // FPS]))
            halves.append((start, start + m - 1))
            start += m
        pxy, bxy, labels = (np.concatenate(p) for p in zip(*parts))
        # Broadcast cutaways (replays, close-ups) track nobody: ~half the dead frames. Our tactical
        # feed never goes blank, so these are ungraded, not free dead-ball points.
        labels[~np.isfinite(pxy[..., 0]).any(1)] = "X"
        players = np.concatenate([pxy, speed(pxy, FPS)[..., None]], -1)
        ball = np.concatenate([bxy, speed(bxy, FPS)[:, None]], -1)
        # halves are stacked back to back, so the 5-frame speed lag reaches across the join
        for s, _ in halves[1:]:
            players[s:s + 5, :, 2] = ball[s:s + 5, 2] = np.nan
        save(f"skillcorner_{mid}", players, ball, team, labels, np.array(halves), pitch, FPS)


if __name__ == "__main__":
    which = sys.argv[1:] or ["idsse", "metrica", "skillcorner"]
    for w in which:
        {"idsse": build_idsse, "metrica": build_metrica, "skillcorner": build_skillcorner}[w]()
