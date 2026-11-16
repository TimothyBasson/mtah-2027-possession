"""Score BAR-ATH submission files against the Smart Tagging over the whole match.

Three partial checks, none of them spellF1 (tags are phases, not possessions, and automatic):
  phase    frames inside a tagged attacking phase (build up, progression, creating chances,
           attacking transition): share of the file's live frames given to the tagged team
  restart  tagged set pieces: is a stoppage drawn at the tag, and does the tagged team restart
  switch   tagged attacking transitions: does the file hand the ball to the tagged team there
Uncertainty: bootstrap over tagged episodes / events, paired against --ref.
Measurement only. Nothing here may be used to fit or tune a model.

python tagcheck/score.py name=path [name=path ...] [--ref name]
"""
import argparse
import json

import numpy as np

from common import ATTACK, FPS, N, PUBLIC, graded, public_gt, read, tags

NEAR = (-3 * FPS, 6 * FPS)       # a stoppage in this stretch around a set-piece tag's start
RESTART = 40 * FPS               # the restart must come within this of the tag's start
SWITCH = (-3 * FPS, 8 * FPS)     # a change to the tagged team in this stretch around a transition
KICK = 85                        # a kick-off tag opens this many frames before the kick
DRAWS = 4000


def phase_frames(T, lo=0, hi=N):
    """tagged team per frame (-1 none or both), and episode id per frame."""
    cov = np.zeros((N, 2), bool)
    for t in T:
        if t["code"] in ATTACK and t["team"] >= 0:
            cov[t["a"]:t["b"] + 1, t["team"]] = True
    team = np.where(cov[:, 0] & ~cov[:, 1], 0, np.where(cov[:, 1] & ~cov[:, 0], 1, -1))
    use = (team >= 0) & graded()
    use[:lo] = False
    use[hi + 1:] = False
    # episodes: maximal runs of use with one team
    edge = np.flatnonzero(np.diff(np.r_[False, use, False].astype(int)) != 0)
    ep = np.full(N, -1)
    k = 0
    for a, b in zip(edge[::2], edge[1::2]):
        cut = a + np.flatnonzero(np.diff(team[a:b]) != 0) + 1
        for x, y in zip(np.r_[a, cut], np.r_[cut, b]):
            ep[x:y] = k
            k += 1
    return team, ep, k


def phase_counts(lab, team, ep, k):
    """per episode: live frames with the tagged team, live frames, all frames."""
    m = ep >= 0
    live = m & (lab < 2)
    ok = np.bincount(ep[live & (lab == team)], minlength=k)
    return ok, np.bincount(ep[live], minlength=k), np.bincount(ep[m], minlength=k)


def restart_events(lab, T):
    """per tagged set piece (no kick-offs): stoppage found, restart team right (nan if no team
    tag or no stoppage)."""
    found, team = [], []
    for t in T:
        if t["code"] != "set pieces" or "kick off" in t["type"]:
            continue
        a = t["a"]
        if not graded()[a]:
            continue
        w = np.flatnonzero(lab[a + NEAR[0]:a + NEAR[1]] == 2)
        found.append(float(len(w) > 0))
        ok = np.nan
        if t["team"] >= 0:
            ok = 0.0
            if len(w):
                i = a + NEAR[0] + w[0]
                nxt = np.flatnonzero(lab[i:a + RESTART] < 2)
                if len(nxt):
                    ok = float(lab[i + nxt[0]] == t["team"])
        team.append(ok)
    return np.array(found), np.array(team)


def switch_events(lab, T):
    out = []
    for t in T:
        if t["code"] != "attacking transition" or t["team"] < 0 or not graded()[t["a"]]:
            continue
        x = lab[t["a"] + SWITCH[0]:t["a"] + SWITCH[1]]
        live = x[x < 2]
        gain = (live[1:] == t["team"]) & (live[:-1] != t["team"])
        out.append(float(gain.any()))
    return np.array(out)


def kickoff_events(lab, T):
    """after each tagged goal: dead for the 15 s before the kick-off, conceding team kicks off."""
    goals = [t for t in T if t["code"] == "goals"]
    dead, team = [], []
    for t in T:
        if t["type"] != "kick off (after goal)":
            continue
        k = t["a"] + KICK
        g = [x for x in goals if abs(x["b"] - t["a"]) <= 2 * FPS]
        dead.append(float(np.mean(lab[k - 15 * FPS:k - 2 * FPS] == 2)))
        nxt = np.flatnonzero(lab[k - 2 * FPS:k + 10 * FPS] < 2)
        team.append(float(len(nxt) > 0 and len(g) == 1 and lab[k - 2 * FPS + nxt[0]] == 1 - g[0]["team"]))
    return np.array(dead), np.array(team)


def boot(stat, n, rng, draws=DRAWS):
    idx = rng.integers(0, n, (draws, n))
    return np.array([stat(i) for i in idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--ref")
    p.add_argument("--json")
    args = p.parse_args()
    files = dict(f.split("=", 1) for f in args.files)
    labs = {k: read(v) for k, v in files.items()}
    ref = args.ref or next(iter(labs))
    T = tags()
    rng = np.random.default_rng(0)

    # how good are the tags? the public five minutes, scored as if they were a file
    gt = public_gt()
    team, ep, k = phase_frames(T, *PUBLIC)
    ok, live, _ = phase_counts(gt, team, ep, k)
    print("Tags against the public five minutes: {} tagged phases, {:.1f} s of live play, the "
          "tagged team has the ball on {:.1%} of it (per phase: {})".format(
              k, live.sum() / FPS, ok.sum() / live.sum(),
              " ".join("{:.0%}".format(a / b) for a, b in zip(ok, live) if b)))
    f, t = restart_events(gt, [x for x in T if PUBLIC[0] <= x["a"] <= PUBLIC[1] - RESTART])
    print("  set pieces fully inside it: {} , stoppage at the tag {}, tagged team restarts {}".format(
        len(f), int(f.sum()), int(np.nansum(t))))
    s = switch_events(gt, [x for x in T if PUBLIC[0] <= x["a"] <= PUBLIC[1]])
    print("  attacking transitions inside it: {}, ball changes to the tagged team {}".format(len(s), int(s.sum())))

    team, ep, k = phase_frames(T)
    print("\nWhole match: {} tagged phase episodes, {:.1f} min, {:.0%} of graded frames".format(
        k, (ep >= 0).sum() / FPS / 60, (ep >= 0).sum() / graded().sum()))
    res = {}
    for name, lab in labs.items():
        ok, live, alln = phase_counts(lab, team, ep, k)
        f, t = restart_events(lab, T)
        s = switch_events(lab, T)
        kd, kt = kickoff_events(lab, T)
        res[name] = dict(ok=ok, live=live, alln=alln, found=f, rteam=t, switch=s, kd=kd, kt=kt)

    def phase_stat(r):
        return lambda i: r["ok"][i].sum() / r["live"][i].sum()

    hdr = "{:<16} {:>22} {:>20} {:>6} {:>18} {:>18} {:>18} {:>9}".format(
        "file", "phase team share", "vs " + ref, "dead", "stoppage at tag", "tagged team restarts",
        "switch at transition", "kick-offs")
    print(hdr)
    out = {}
    r0 = res[ref]
    tm = ~np.isnan(r0["rteam"])
    for name, r in res.items():
        b = boot(phase_stat(r), k, np.random.default_rng(1))
        b0 = boot(phase_stat(r0), k, np.random.default_rng(1))
        d = b - b0
        v = r["ok"].sum() / r["live"].sum()
        nf, nt, ns = len(r["found"]), int(tm.sum()), len(r["switch"])

        def ev(x):
            bb = boot(lambda i: np.mean(x[i]), len(x), np.random.default_rng(2))
            return "{:>2.0f}/{} [{:.0%},{:.0%}]".format(x.sum(), len(x), *np.percentile(bb, [2.5, 97.5]))
        row = dict(phase=v, phase_ci=list(np.percentile(b, [2.5, 97.5])),
                   diff=float(v - r0["ok"].sum() / r0["live"].sum()), diff_ci=list(np.percentile(d, [2.5, 97.5])),
                   p_better=float(np.mean(d > 0)),
                   dead=float(1 - r["live"].sum() / r["alln"].sum()),
                   found=int(r["found"].sum()), n_found=nf, rteam=int(np.nansum(r["rteam"])), n_rteam=nt,
                   switch=int(r["switch"].sum()), n_switch=ns,
                   kick_dead=float(r["kd"].mean()), kick_team=int(r["kt"].sum()))
        out[name] = row
        print("{:<16} {:.3f} [{:.3f},{:.3f}]  {:+.3f} [{:+.3f},{:+.3f}] {:>5.0%} {:>18} {:>18} {:>18} {:>4.0%} {}/{}".format(
            name, v, *row["phase_ci"], row["diff"], *row["diff_ci"], row["dead"],
            ev(r["found"]), ev(r["rteam"][tm]), ev(r["switch"]), row["kick_dead"], row["kick_team"], len(r["kt"])))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
