"""Possession story timeline and turnover autopsy for BAR-ATH, as one self-contained HTML page.

    python story/make_story.py                                  # -> data/viz/story.html
    python story/make_story.py --labels my_labels.csv --out /tmp/story.html

Reads the tracking (possession/load.py, keepers corrected), a frame,label possession file (default:
the file we submitted, /mnt/project-files/final/Gambtt_submitted.csv, or data/final/ on a laptop)
and the Smart Tagging XML. Everything the page needs is written into it: phases, turnovers,
role templates, and the tracking at 5 Hz (gzip, decoded in the browser).

What it computes (details in story/README.md):
- phases: maximal runs of H or A in the labels; how each was won (open play or which restart) and
  lost (turnover or ball out), territory gained, and the Smart Tagging moments inside it
- roles: per team, an average shape in and out of possession, learned by matching players to
  slots frame by frame (tracker slots are not players, so roles stand in for names)
- turnovers: every open-play change of team. Shape 5 s before against the team's average, the
  roles furthest from their usual spot, the action before the loss, and what happened next
  (counter-press, time to start retreating, ball won back, opponent reaching the final third)
"""

import argparse
import base64
import gzip
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "possession"))
import load  # noqa: E402
import keepers as KP  # noqa: E402
from games import require  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "submission"))
from make_submission import fill_short_abstentions  # noqa: E402

FPS = 25
STEP = 5                       # tracking sent to the page every 5 frames (5 Hz)
L, W = load.PITCH
TEAMS = ["Barcelona", "Athletic Club"]
TAG_TEAM = {"Barcelona": 0, "Atletic Club": 1}
BEFORE = 5 * FPS               # the autopsy looks 5 s before the loss
ATTACK_TAGS = {"BUILD UP": "build-up", "PROGRESSION": "progression", "BALL IN FINAL THIRD": "final third",
               "BALL IN THE BOX": "ball in the box", "CREATING CHANCES": "chance created",
               "GOALS": "GOAL", "LONG BALL": "long ball", "ATTACKING TRANSITION": "attacking transition"}


def default_labels():
    for p in ("/mnt/project-files/final/Gambtt_submitted.csv", os.path.join(ROOT, "data", "final", "Gambtt_submitted.csv")):
        if os.path.exists(p):
            return p
    return None


def runs_of(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((lab[s], s, i - 1))
            s = i
    return out


def read_tags(path):
    tags = []
    for inst in ET.parse(path).getroot().iter("instance"):
        d = {"code": inst.findtext("code"), "a": float(inst.findtext("start")), "b": float(inst.findtext("end"))}
        for lab in inst.iter("label"):
            d[lab.findtext("group")] = lab.findtext("text")
        tags.append(d)
    return tags


class Match:
    def __init__(self, labels_path):
        self.players, self.ball, self.team, self.halves = load.load()
        self.keepers = set(KP.keeper_slots(require(["atd_meta"])[0])) | {len(self.team) - 2, len(self.team) - 1}
        self.xy = self.players[..., :2]
        self.bxy = self.ball[:, :2]
        n = len(self.xy)
        lab = np.loadtxt(labels_path, dtype=str, delimiter=",", skiprows=1)[:, 1]
        assert len(lab) == n, (len(lab), n)
        # short X gaps carry the previous label forward, as the scorer does; long ones (before
        # kick-off, half-time, after the whistle) stay breaks
        self.lab = np.array(fill_short_abstentions(list(lab), 60 * FPS))
        self.outfield = np.array([k not in self.keepers for k in range(len(self.team))])
        # attack sign per team per frame: +1 if the team attacks towards x = L
        self.half_of = np.full(n, -1)
        self.sign = np.zeros((n, 2))
        for i, (a, b) in enumerate(self.halves):
            self.half_of[a:b + 1] = i
            mx = [np.nanmean(self.xy[a:b + 1][:, self.outfield & (self.team == k), 0]) for k in (0, 1)]
            home_low = mx[0] < mx[1]
            self.sign[a:b + 1, 0] = 1 if home_low else -1
            self.sign[a:b + 1, 1] = -self.sign[a:b + 1, 0]

    def team_xy(self, k, f, outfield=True):
        m = (self.team == k) & (self.outfield if outfield else True)
        p = self.xy[f][m]
        return p[np.isfinite(p[:, 0])]

    def ball_at(self, f, win=FPS):
        """Ball position at f, or the nearest seen within win frames, else None."""
        n = len(self.bxy)
        for d in range(0, win + 1):
            for g in (f - d, f + d):
                if 0 <= g < n and np.isfinite(self.bxy[g, 0]):
                    return self.bxy[g], abs(d)
        return None, None

    def contest_point(self, f, loser):
        """Where the ball probably was: the ball if tracked within 1 s, else the midpoint of the
        closest pair of opposing players."""
        b, _ = self.ball_at(f)
        if b is not None:
            return b, True
        a, c = self.team_xy(loser, f), self.team_xy(1 - loser, f)
        if len(a) == 0 or len(c) == 0:
            return None, False
        d = np.linalg.norm(a[:, None] - c[None], axis=-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return (a[i] + c[j]) / 2, False


LEFT_IS_LOW_Y = True   # set by orientation_check()


def orientation_check(m, tags):
    """Smart Tagging marks the side of final-third entries. Which y is the attacking team's left?"""
    global LEFT_IS_LOW_Y
    votes = Counter()
    for t in tags:
        if t["code"] != "BALL IN FINAL THIRD" or t.get("Side") not in ("Left", "Right") or t.get("Team") not in TAG_TEAM:
            continue
        k = TAG_TEAM[t["Team"]]
        a, b = int(t["a"] * FPS), int(t["b"] * FPS)
        ys = m.bxy[a:b, 1]
        ys = ys[np.isfinite(ys)]
        if len(ys) < 10:
            continue
        s = m.sign[(a + b) // 2, k]
        if s == 0:
            continue
        # facing +x with y growing down the screen, left is low y
        low_y_is_left = (np.median(ys) < W / 2) if s > 0 else (np.median(ys) > W / 2)
        votes[(t["Side"], low_y_is_left)] += 1
    agree = votes[("Left", True)] + votes[("Right", False)]
    disagree = votes[("Left", False)] + votes[("Right", True)]
    LEFT_IS_LOW_Y = agree >= disagree
    return agree, disagree


def to_att(m, xy, k, f):
    """(..., 2) pitch metres -> team k's attacking frame at f: x metres from own goal line, y metres
    left of the centre line (positive = left, looking up the pitch)."""
    s = m.sign[f, k]
    x = xy[..., 0] if s > 0 else L - xy[..., 0]
    yc = xy[..., 1] - W / 2
    # facing +x: left is low y if LEFT_IS_LOW_Y. Facing -x it is the other way
    left = -yc if (s > 0) == LEFT_IS_LOW_Y else yc
    return np.stack([x, left], -1)


def zone_names(p):
    x, y = p
    third = "own third" if x < L / 3 else ("middle third" if x < 2 * L / 3 else "final third")
    lane = "left" if y > W / 6 else ("right" if y < -W / 6 else "centre")
    return third, lane


# ---------------------------------------------------------------------------------------------
# role templates
# ---------------------------------------------------------------------------------------------

def rel_positions(m, k, f):
    """Outfield players of team k at f, relative to their centroid, in k's attacking frame."""
    p = m.team_xy(k, f)
    if len(p) < 7:
        return None, None
    a = to_att(m, p, k, f)
    c = a.mean(0)
    return a - c, c


def fit_template(samples, n_roles=10, iters=8):
    full = [s for s in samples if len(s) == n_roles]
    allp = np.concatenate(full)
    # start: sort a median frame's players by depth then width
    rng = np.random.default_rng(0)
    T = full[rng.integers(len(full))].copy()
    for _ in range(iters):
        acc = np.zeros_like(T)
        cnt = np.zeros(n_roles)
        for s in samples:
            r, c = linear_sum_assignment(np.linalg.norm(s[:, None] - T[None], axis=-1))
            acc[c] += s[r]
            cnt[c] += 1
        T = acc / np.maximum(cnt, 1)[:, None]
    # typical miss per role
    devs = defaultdict(list)
    for s in samples:
        r, c = linear_sum_assignment(np.linalg.norm(s[:, None] - T[None], axis=-1))
        for i, j in zip(r, c):
            devs[j].append(np.linalg.norm(s[i] - T[j]))
    spread = np.array([np.percentile(devs[j], 90) if devs[j] else 10 for j in range(n_roles)])
    return T, spread


def name_roles(T):
    x = T[:, 0]
    order = np.argsort(x)
    # lines: deepest 4 defenders, top 2 forwards, rest midfield (a 4-4-2 / 4-3-3 reading, good enough for names)
    line = np.empty(len(T), object)
    line[order[:4]] = "back"
    line[order[-2:]] = "forward"
    for j in order[4:-2]:
        line[j] = "mid"
    names = [None] * len(T)
    for ln in ("back", "mid", "forward"):
        idx = [j for j in range(len(T)) if line[j] == ln]
        idx.sort(key=lambda j: -T[j, 1])            # left (positive) first
        if ln == "back":
            labels = {4: ["left-back", "left centre-back", "right centre-back", "right-back"]}.get(len(idx))
        elif ln == "forward":
            labels = ["left forward", "right forward"] if len(idx) == 2 else None
        else:
            labels = {4: ["left midfielder", "left-centre mid", "right-centre mid", "right midfielder"],
                      3: ["left midfielder", "central midfielder", "right midfielder"]}.get(len(idx))
        if labels is None:
            labels = [f"{ln} {i + 1}" for i in range(len(idx))]
        for j, nm in zip(idx, labels):
            names[j] = nm
    return names


def build_templates(m, phases):
    tpl = {}
    for k in (0, 1):
        for state in ("in", "out"):
            samples = []
            for ph in phases:
                own = ph["team"] == k
                if own != (state == "in"):
                    continue
                for f in range(ph["a"] + FPS, ph["b"] - FPS, 10):
                    r, _ = rel_positions(m, k, f)
                    if r is not None and len(r) <= 10:
                        samples.append(r)
            T, spread = fit_template(samples)
            tpl[(k, state)] = dict(T=T, spread=spread, n=len(samples))
        # football names come from the shape without the ball, where a back four and lines are
        # clear. The roles with the ball get the name of the role the same player holds without it:
        # tracker slots keep a player for a while, so across each open-play turnover we count which
        # with-ball role a slot held 2 s before and which without-ball role it held 4 s after (or
        # the other way round), and match the two sets of roles on those counts.
        tpl[(k, "out")]["names"] = name_roles(tpl[(k, "out")]["T"])
        co = np.zeros((10, 10))
        for i in range(len(phases) - 1):
            p, q = phases[i], phases[i + 1]
            if p["lost"] != "open" or q["team"] == p["team"]:
                continue
            f = p["b"]
            a_state, b_state = ("in", "out") if p["team"] == k else ("out", "in")
            ra, _ = assign(m, k, f - 2 * FPS, tpl[(k, a_state)])
            rb, _ = assign(m, k, f + 4 * FPS, tpl[(k, b_state)])
            if not ra or not rb:
                continue
            da = {slot: j for slot, j, _ in ra}
            for slot, j, _ in rb:
                if slot in da:
                    jin, jout = (da[slot], j) if a_state == "in" else (j, da[slot])
                    co[jin, jout] += 1
        r, c = linear_sum_assignment(-co)
        names_in = [None] * 10
        for jin, jout in zip(r, c):
            names_in[jin] = tpl[(k, "out")]["names"][jout]
        tpl[(k, "in")]["names"] = names_in
        tpl[(k, "in")]["link_agree"] = float(co[r, c].sum() / max(co.sum(), 1))
    return tpl


def assign(m, k, f, tpl):
    """Which slot plays which role at f: list of (slot, role, deviation vector) plus centroid."""
    mask = (m.team == k) & m.outfield
    slots = np.flatnonzero(mask)
    p = m.xy[f][slots]
    ok = np.isfinite(p[:, 0])
    slots, p = slots[ok], p[ok]
    if len(p) < 7:
        return None, None
    a = to_att(m, p, k, f)
    c = a.mean(0)
    rel = a - c
    T = tpl["T"]
    r, cidx = linear_sum_assignment(np.linalg.norm(rel[:, None] - T[None], axis=-1))
    return [(int(slots[i]), int(j), rel[i] - T[j]) for i, j in zip(r, cidx)], c


# ---------------------------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------------------------

def overlapping_tags(tags, a, b, pad=0.0):
    ta, tb = a / FPS - pad, b / FPS + pad
    return [t for t in tags if t["b"] >= ta and t["a"] <= tb]


def build_phases(m, tags):
    runs = runs_of(m.lab)
    phases = []
    for i, (lb, a, b) in enumerate(runs):
        if lb not in "HA" or m.half_of[a] < 0:
            continue
        k = 0 if lb == "H" else 1
        prev = runs[i - 1] if i > 0 else None
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        won = "kick-off" if prev is None else ("open" if prev[0] in "HA" else "restart")
        lost = "end" if nxt is None else ("open" if nxt[0] in "HA" else ("stoppage" if nxt[0] == "D" else "end"))
        restart = None
        if won == "restart":
            for t in overlapping_tags(tags, a - 3 * FPS, a + 2 * FPS):
                if t["code"] == "SET PIECES" and t.get("Type"):
                    restart = t["Type"].replace(" (Start)", "").replace(" (After Goal)", " after a goal")
                    break
        st, sb = m.contest_point(a, 1 - k)
        en, eb = m.contest_point(b, k)
        sa = to_att(m, st, k, a) if st is not None else None
        ea = to_att(m, en, k, b) if en is not None else None
        # furthest the ball got (seen frames only), in k's attack frame
        bb = m.bxy[a:b + 1]
        okb = np.isfinite(bb[:, 0])
        maxx = float(to_att(m, bb[okb], k, a)[:, 0].max()) if okb.sum() > 5 else None
        moments = []
        for t in overlapping_tags(tags, a, b):
            if t["code"] in ATTACK_TAGS and TAG_TEAM.get(t.get("Team")) == k:
                if ATTACK_TAGS[t["code"]] not in moments:
                    moments.append(ATTACK_TAGS[t["code"]])
        phases.append(dict(i=len(phases), team=k, a=int(a), b=int(b), dur=round((b - a + 1) / FPS, 1),
                           half=int(m.half_of[a]), won=won, lost=lost, restart=restart,
                           start=None if sa is None else [round(float(v), 1) for v in sa],
                           end=None if ea is None else [round(float(v), 1) for v in ea],
                           start_ball=bool(sb), end_ball=bool(eb),
                           maxx=None if maxx is None else round(maxx, 1),
                           ball_seen=round(float(okb.mean()), 2), moments=moments))
    return phases


def clock(m, f):
    h = m.half_of[f]
    a = m.halves[h][0] if h >= 0 else 0
    s = (f - a) / FPS + (45 * 60 if h == 1 else 0)
    return f"{int(s // 60)}'{int(s % 60):02d}"


def where(p):
    if p is None:
        return None
    third, lane = zone_names(p)
    third = "their own third" if third == "own third" else f"the {third}"
    return third + ("" if lane == "centre" else f" on the {lane}")


def summarise(m, ph, phases):
    T = TEAMS[ph["team"]]
    O = TEAMS[1 - ph["team"]]
    z0, z1 = where(ph["start"]), where(ph["end"])
    if ph["won"] == "open":
        s = f"{T} won the ball in open play" + (f" in {z0}" if z0 else "")
    elif ph["won"] == "restart":
        s = f"{T} restarted" + (f" with a {ph['restart'].lower()}" if ph["restart"] else " after a stoppage") + (f" in {z0}" if z0 else "")
    else:
        s = f"{T} kicked off"
    s0 = ph["start"][0] if ph["start"] else None
    if ph["maxx"] is not None and s0 is not None:
        gain = ph["maxx"] - s0
        if ph["maxx"] > L - 16.5:
            s += f" and got the ball into the box ({gain:+.0f} m up the pitch)"
        elif ph["maxx"] > 2 * L / 3:
            s += f" and reached the final third ({gain:+.0f} m)"
        elif gain > 10:
            s += f" and moved it {gain:.0f} m forward"
        else:
            s += " but made little ground"
    s += f" in {ph['dur']:.0f} s."
    if ph["lost"] == "open":
        s += f" Lost in open play" + (f" in {z1}" if z1 else "") + f"; {O} took over."
    elif ph["lost"] == "stoppage":
        s += " Ended with the ball going dead" + (f" in {z1}" if z1 else "") + "."
    if ph["moments"]:
        s += " Tagged: " + ", ".join(ph["moments"]) + "."
    if ph["dur"] < 3:
        s += " A brief spell: the model is least sure about these."
    return s


# ---------------------------------------------------------------------------------------------
# turnover autopsy
# ---------------------------------------------------------------------------------------------

def shape(m, k, f):
    p = m.team_xy(k, f)
    if len(p) < 7:
        return None
    a = to_att(m, p, k, f)
    return dict(height=float(a[:, 0].mean()), width=float(a[:, 1].max() - a[:, 1].min()),
                depth=float(a[:, 0].max() - a[:, 0].min()), last_line=float(np.sort(a[:, 0])[:2].mean()))


def preceding_action(m, k, f, tags):
    """What the ball did in the 2 s before the loss, in the loser's attacking frame."""
    for t in overlapping_tags(tags, f - 2 * FPS, f):
        if t["code"] == "LONG BALL" and TAG_TEAM.get(t.get("Team")) == k:
            return "long ball"
    b1, d1 = m.ball_at(f, win=12)
    b0, d0 = m.ball_at(f - 2 * FPS, win=12)
    if b1 is None or b0 is None:
        return "ball not tracked"
    v = to_att(m, b1, k, f) - to_att(m, b0, k, f)
    dx, dy = float(v[0]), float(abs(v[1]))
    if dx < -5:
        return "back pass"
    if dx > 20:
        return "long forward ball"
    if dx > 6:
        return "forward pass"
    if dy > 10:
        return "square ball"
    return "carry or short pass"


def reaction(m, k, f, phases_by_start, tags):
    """After team k loses it at f: players within 10 m of the ball at +2 s, seconds until the team
    is moving back towards its own goal, whether it won the ball back within 5 s / 10 s, and whether
    the opponent reached k's own third."""
    out = {}
    b, _ = m.ball_at(f + 2 * FPS, win=12)
    if b is None:
        b, _ = m.contest_point(f, k)
    p = m.team_xy(k, f + 2 * FPS)
    out["press10"] = int((np.linalg.norm(p - b, axis=-1) < 10).sum()) if b is not None and len(p) else None
    # centroid x in k's frame, smoothed over 1 s; first time its 1 s velocity is below -0.5 m/s
    cx = []
    for g in range(f, min(f + 10 * FPS, len(m.xy))):
        q = m.team_xy(k, g)
        cx.append(to_att(m, q, k, f)[:, 0].mean() if len(q) >= 7 else np.nan)
    cx = np.array(cx)
    rt = None
    for g in range(FPS, len(cx)):
        v = (cx[g] - cx[g - FPS])
        if np.isfinite(v) and v < -0.5:
            rt = round((g - FPS / 2) / FPS, 1)
            break
    out["retreat_s"] = rt
    return out


def build_turnovers(m, phases, tpl, tags):
    avg_shape = {}
    for k in (0, 1):
        ss = [shape(m, k, f) for ph in phases if ph["team"] == k for f in range(ph["a"], ph["b"], 25)]
        ss = [s for s in ss if s]
        avg_shape[k] = {key: (float(np.mean([s[key] for s in ss])), float(np.std([s[key] for s in ss]))) for key in ss[0]}
    tos = []
    for idx, ph in enumerate(phases):
        if ph["lost"] != "open":
            continue
        k = ph["team"]
        f = ph["b"]
        loc, seen = m.contest_point(f, k)
        if loc is None:
            continue
        la = to_att(m, loc, k, f)
        third, lane = zone_names(la)
        g = max(ph["a"] - 0, f - BEFORE)            # 5 s before, or the start of a shorter spell
        pre_state = "in" if g >= ph["a"] else "out"
        roles, cen = assign(m, k, g, tpl[(k, pre_state)])
        flagged = []
        if roles:
            T = tpl[(k, pre_state)]
            for slot, j, dv in roles:
                d = float(np.linalg.norm(dv))
                z = d / T["spread"][j]
                if z > 1.0 and d > 8:
                    dirs = []
                    if abs(dv[0]) > 4:
                        dirs.append(f"{abs(dv[0]):.0f} m {'higher' if dv[0] > 0 else 'deeper'}")
                    if abs(dv[1]) > 4:
                        dirs.append(f"{abs(dv[1]):.0f} m {'further left' if dv[1] > 0 else 'further right'}")
                    flagged.append(dict(slot=slot, role=T["names"][j], dist=round(d, 1), z=round(z, 2),
                                        how=" and ".join(dirs) or f"{d:.0f} m off"))
            flagged.sort(key=lambda r: -r["z"])
        sh = shape(m, k, g)
        notes = []
        if sh:
            for key, word_hi, word_lo in (("height", "pushed up", "sitting deep"), ("width", "stretched wide", "narrow"),
                                          ("depth", "long (lines far apart)", "compact")):
                mu, sd = avg_shape[k][key]
                zz = (sh[key] - mu) / sd
                if zz > 1:
                    notes.append(word_hi)
                elif zz < -1:
                    notes.append(word_lo)
        # behind the ball at the moment of loss (rest defence) vs opponents ahead of it
        p_l = m.team_xy(k, f)
        p_w = m.team_xy(1 - k, f)
        behind = int((to_att(m, p_l, k, f)[:, 0] < la[0]).sum()) if len(p_l) else None
        attackers = int((to_att(m, p_w, k, f)[:, 0] < la[0]).sum()) if len(p_w) else None
        act = preceding_action(m, k, f, tags)
        rx = reaction(m, k, f, None, tags)
        # what happened next: the opponent's spell
        nxt = phases[idx + 1] if idx + 1 < len(phases) else None
        back5 = back10 = False
        danger = False
        if nxt is not None and nxt["team"] == 1 - k:
            danger = (nxt["maxx"] is not None and nxt["maxx"] > 2 * L / 3) or any(
                mm in ("chance created", "GOAL", "ball in the box") for mm in nxt["moments"])
            if nxt["lost"] == "open":
                back5 = nxt["dur"] <= 5
                back10 = nxt["dur"] <= 10
        quick = ph["dur"] < 5
        pattern = f"lost in the {third}, {lane} side, after a {act}" if act != "ball not tracked" else f"lost in the {third}, {lane} side"
        tos.append(dict(i=len(tos), phase=ph["i"], team=k, f=int(f), at=[round(float(v), 1) for v in la],
                        ball_seen=bool(seen), third=third, lane=lane, action=act, pattern=pattern,
                        before=int(g), before_s=round((f - g) / FPS, 1), pre_state=pre_state,
                        flagged=flagged[:3], shape=None if sh is None else {kk: round(v, 1) for kk, v in sh.items()},
                        shape_notes=notes, behind=behind, attackers_ahead=attackers, quick_loss=quick,
                        press10=rx["press10"], retreat_s=rx["retreat_s"], won_back5=back5, won_back10=back10,
                        danger=bool(danger), clock=clock(m, f),
                        roles_before=[[slot, j] for slot, j, _ in roles] if roles else []))
    return tos, avg_shape


def rank_patterns(tos):
    out = []
    groups = defaultdict(list)
    for t in tos:
        groups[(t["team"], t["third"], t["lane"])].append(t)
        if t["action"] != "ball not tracked":
            groups[(t["team"], t["third"], t["lane"], t["action"])].append(t)
        groups[(t["team"], "action", t["action"])].append(t)
        if t["quick_loss"]:
            groups[(t["team"], "quick")].append(t)
        for nt in t["shape_notes"]:
            groups[(t["team"], "shape", nt)].append(t)
        for fl in t["flagged"][:1]:
            groups[(t["team"], "role", fl["role"])].append(t)
    for key, ts in groups.items():
        k = key[0]
        if len(ts) < 3:
            continue
        if key[1] == "action":
            if key[2] == "ball not tracked":
                continue
            text = f"after a {key[2]}"
        elif key[1] == "quick":
            text = "within 5 s of winning it"
        elif key[1] == "shape":
            text = f"with the team {key[2]} 5 s before"
        elif key[1] == "role":
            text = f"with the {key[2]} out of position 5 s before"
        elif len(key) == 4:
            text = f"in the {key[1]}, {key[2]} side, after a {key[3]}"
        else:
            text = f"in the {key[1]}, {key[2]} side"
        n = len(ts)
        out.append(dict(team=k, text=text, n=n, ids=[t["i"] for t in ts],
                        danger=round(sum(t["danger"] for t in ts) / n, 2),
                        back5=round(sum(t["won_back5"] for t in ts) / n, 2),
                        specific=len(key)))
    out.sort(key=lambda r: (-r["n"], -r["specific"]))
    return out


def tracking_blob(m):
    """uint8 x, y per slot every STEP frames inside the halves (255 = missing), gzip, base64."""
    frames = np.concatenate([np.arange(a, b + 1, STEP) for a, b in m.halves])
    xy = np.concatenate([m.xy, m.bxy[:, None]], 1)[frames]
    q = np.full(xy.shape, 255, np.uint8)
    ok = np.isfinite(xy)
    sx = np.clip(np.round(xy[..., 0] / L * 250), 0, 250)
    sy = np.clip(np.round(xy[..., 1] / W * 250), 0, 250)
    q[..., 0] = np.where(ok[..., 0], sx, 255)
    q[..., 1] = np.where(ok[..., 1], sy, 255)
    return base64.b64encode(gzip.compress(q.tobytes(), 9)).decode(), frames, xy.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=default_labels())
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "viz", "story.html"))
    ap.add_argument("--json", default=None, help="also write the analysis as JSON here")
    ap.add_argument("--no-tracking", action="store_true",
                    help="leave the tracking out (no replay): the version that may be shared, since RULES.md "
                         "forbids publishing frame-level tracking exports")
    args = ap.parse_args()
    if not args.labels or not os.path.exists(args.labels):
        sys.exit("no labels file: pass --labels <frame,label csv> (the default is final/Gambtt_submitted.csv "
                 "in /mnt/project-files or data/final/)")
    m = Match(args.labels)
    tags = read_tags(require(["smart_tagging"])[0])
    agree, disagree = orientation_check(m, tags)
    print(f"left/right check on Smart Tagging sides: {agree} agree, {disagree} disagree; left is low y: {LEFT_IS_LOW_Y}")
    phases = build_phases(m, tags)
    tpl = build_templates(m, phases)
    for ph in phases:
        ph["clock"] = clock(m, ph["a"])
        ph["summary"] = summarise(m, ph, phases)
    tos, avg_shape = build_turnovers(m, phases, tpl, tags)
    pats = rank_patterns(tos)
    blob, frames, nslots = tracking_blob(m)
    if args.no_tracking:
        blob = ""
    halves_steps = [[int(a), int(b)] for a, b in m.halves]
    tags_out = [dict(code=t["code"], a=t["a"], b=t["b"], team=TAG_TEAM.get(t.get("Team"), -1),
                     type=t.get("Type")) for t in tags]
    # pitch-frame role templates for drawing ghosts: T relative to centroid in attacking frame
    tpl_out = {f"{k}_{s}": dict(T=np.round(v["T"], 2).tolist(), names=v["names"], spread=np.round(v["spread"], 1).tolist(), n=v["n"])
               for (k, s), v in tpl.items()}
    data = dict(teams=TEAMS, L=L, W=W, fps=FPS, step=STEP, halves=halves_steps, nslots=nslots,
                team=m.team.tolist(), keepers=sorted(int(k) for k in m.keepers), n_frames=len(m.xy),
                sign=[[int(m.sign[a, 0]), int(m.sign[a, 1])] for a, b in m.halves], left_is_low_y=LEFT_IS_LOW_Y,
                phases=phases, turnovers=tos, patterns=pats, templates=tpl_out, tags=tags_out,
                avg_shape={str(k): v for k, v in avg_shape.items()},
                labels_file=os.path.basename(args.labels))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(data, fh, indent=1)
    with open(os.path.join(HERE, "story_template.html")) as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))).replace("__TRACK__", blob)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)
    n_open = sum(p["lost"] == "open" for p in phases)
    print(f"{len(phases)} phases, {n_open} lost in open play, {len(tos)} turnovers analysed, {len(pats)} patterns")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    for k in (0, 1):
        print(TEAMS[k], [p["text"] + f" ({p['n']})" for p in pats if p["team"] == k][:6])


if __name__ == "__main__":
    main()
