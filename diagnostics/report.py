"""Print the diagnostics tables from analyze.py's JSON.   python diagnostics/report.py main [team]

Every mean is the held-out mean external/heldout.py reports: the three providers counted equally."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from analyze import DUR_NAMES  # noqa: E402

PROV = ["idsse", "metrica", "skillcorner"]


def load(tag):
    return json.load(open(os.path.join(lib.ROOT, "data", "cache", "diag", tag + ".json")))


def by_prov(rows, f):
    """[idsse, metrica, skillcorner, equal-weight mean] of f(row) averaged over matches."""
    v = [np.mean([f(r) for r in rows if r["provider"] == p], 0) for p in PROV]
    return v + [np.mean(v, 0)]


def line(label, vals, fmt="{:7.3f}"):
    print(f"{label:44s}" + " ".join(fmt.format(x) for x in vals))


def report(tag):
    rows = load(tag)
    print(f"\n================ {tag} ================")
    print(f"{'':44s}  idsse metrica skillc.   mean")
    for k in ("spellF1", "insample", "macroF1", "alive_dead", "team_acc", "team_acc_live"):
        line(k, by_prov(rows, lambda r: r[k]))
    line("possessions: true", by_prov(rows, lambda r: r["n_gt"]), "{:7.0f}")
    line("possessions: predicted", by_prov(rows, lambda r: r["n_pred"]), "{:7.0f}")
    line("dead share: true", by_prov(rows, lambda r: r["dead_share_true"]))
    line("dead share: predicted", by_prov(rows, lambda r: r["dead_share_pred"]))
    for i in ("0.3", "0.5", "0.7"):
        line(f"IoU {i}: precision", by_prov(rows, lambda r: r["per_iou"][i][0]))
        line(f"IoU {i}: recall", by_prov(rows, lambda r: r["per_iou"][i][1]))
    print("\n-- spread between matches --")
    for p in PROV:
        s = np.array([r["spellF1"] for r in rows if r["provider"] == p])
        b = np.mean([(r["boot"][1] - r["boot"][0]) / 2 for r in rows if r["provider"] == p])
        print(f"{p:12s} n={len(s):2d} mean {s.mean():.3f} sd {s.std(ddof=1):.3f} min {s.min():.3f} max {s.max():.3f}"
              f"  se of mean {s.std(ddof=1) / np.sqrt(len(s)):.3f}  one match's own 5-95% half-width {b:.3f}")
    se = np.sqrt(sum(np.var([r["spellF1"] for r in rows if r["provider"] == p], ddof=1) /
                     sum(r["provider"] == p for r in rows) for p in PROV)) / 3
    print(f"se of the three-provider mean: {se:.4f}")

    print("\n-- score with one thing repaired from the truth (gain over the model) --")
    base = by_prov(rows, lambda r: r["spellF1"])
    keys = list(rows[0]["repairs"])
    table = [(k, by_prov(rows, lambda r: r["repairs"][k])) for k in keys]
    for k, v in table:
        if k.startswith("truth_after") or "__" in k and not k.startswith(("true_", "model_")):
            continue
        line(k, list(np.array(v) - np.array(base)) + [v[3]], "{:+7.3f}")
    print("\n-- the model's own clean-up applied to the truth (ceiling) --")
    for k, v in table:
        if k.startswith("truth_after"):
            line(k, v)
    print("\n-- with the other half already right: fix one class (gain), or leave only that class wrong (loss from 1.0) --")
    for k, v in table:
        if k.startswith("fix_") and (k.endswith("__true_dead") or k.endswith("__true_team")):
            ref = by_prov(rows, lambda r: r["repairs"]["true_dead__model_team" if k.endswith("__true_dead") else "model_dead__true_team"])
            line(k, list(np.array(v) - np.array(ref)), "{:+7.3f}")
    for k, v in table:
        if k.startswith("only_"):
            line(k, list(np.array(v) - 1), "{:+7.3f}")
    print("\n-- share of live frames (team classes) / graded frames (dead classes) in each class of mistake --")
    for k in rows[0]["error_frames"]:
        den = "graded_frames" if "dead" in k else "live_frames"
        line(k, by_prov(rows, lambda r: r["error_frames"][k] / r[den]))

    print("\n-- recall by true possession length (share of true possessions; mean recall over the 5 IoUs) --")
    for i, nm in enumerate(DUR_NAMES):
        sh = by_prov(rows, lambda r: r["recall_by_dur"][i][0] / r["n_gt"])
        rc = [sum(r["recall_by_dur"][i][1] for r in rows if r["provider"] == p) /
              max(1, sum(r["recall_by_dur"][i][0] for r in rows if r["provider"] == p)) for p in PROV]
        print(f"{nm:8s} share " + " ".join(f"{x:5.2f}" for x in sh[:3]) + "   recall " + " ".join(f"{x:5.2f}" for x in rc + [np.mean(rc)]))
    print("-- precision by predicted possession length (share of predicted; mean precision) --")
    for i, nm in enumerate(DUR_NAMES):
        sh = by_prov(rows, lambda r: r["precision_by_dur"][i][0] / max(1, r["n_pred"]))
        rc = [sum(r["precision_by_dur"][i][1] for r in rows if r["provider"] == p) /
              max(1, sum(r["precision_by_dur"][i][0] for r in rows if r["provider"] == p)) for p in PROV]
        print(f"{nm:8s} share " + " ".join(f"{x:5.2f}" for x in sh[:3]) + "   precis " + " ".join(f"{x:5.2f}" for x in rc + [np.mean(rc)]))
    print("\n-- recall by how the true possession starts>ends (share, recall) --")
    for c in sorted({c for r in rows for c in r["recall_by_context"]}):
        out = []
        for p in PROV:
            n = sum(r["recall_by_context"].get(c, [0, 0])[0] for r in rows if r["provider"] == p)
            m = sum(r["recall_by_context"].get(c, [0, 0])[1] for r in rows if r["provider"] == p)
            tot = sum(r["n_gt"] for r in rows if r["provider"] == p)
            out.append(f"{n / tot:5.2f} {m / max(1, n):5.2f}")
        print(f"{c:20s} " + "   ".join(out))
    print("\n-- true possessions with no match at IoU 0.5: what the prediction shows there (share of all true possessions) --")
    line("matched at 0.5", by_prov(rows, lambda r: r["n_matched_0.5"] / r["n_gt"]))
    for c in sorted({c for r in rows for c in r["unmatched_at_0.5"]}):
        line(c, by_prov(rows, lambda r: r["unmatched_at_0.5"].get(c, 0) / r["n_gt"]))

    print("\n-- timing, seconds (model minus truth; + is late) --")
    for k, nm in (("turnover_offsets", "turnovers"), ("stop_offsets", "stoppage starts"), ("restart_offsets", "restarts")):
        for p in PROV:
            o = np.concatenate([np.array(r[k], float) for r in rows if r["provider"] == p]) / lib.FPS
            f = o[np.isfinite(o)]
            print(f"{nm:16s} {p:12s} found within 5 s {np.isfinite(o).mean():.2f}  median {np.median(f):+.2f}"
                  f"  25-75% {np.percentile(f, 25):+.2f}..{np.percentile(f, 75):+.2f}"
                  f"  within 0.5 s {np.mean(np.abs(o) <= .5):.2f}  1 s {np.mean(np.abs(o) <= 1):.2f}  2 s {np.mean(np.abs(o) <= 2):.2f}")
    line("true turnovers per match", by_prov(rows, lambda r: len(r["turnover_offsets"])), "{:7.0f}")
    line("model team flips per match", by_prov(rows, lambda r: r["pred_flips"]), "{:7.0f}")
    line("  of them near no true turnover", by_prov(rows, lambda r: r["false_flips"]), "{:7.0f}")
    line("true stoppages per match", by_prov(rows, lambda r: len(r["stop_offsets"])), "{:7.0f}")
    line("model stoppages per match", by_prov(rows, lambda r: r["pred_stops"]), "{:7.0f}")
    line("  of them near no true stoppage start", by_prov(rows, lambda r: r["false_stops"]), "{:7.0f}")
    print("\n-- whole prediction slid in time (frames: spellF1) --")
    ks = list(rows[0]["shift"])
    print("frames      " + " ".join(f"{int(k):+6d}" for k in ks))
    for nm in ("shift", "shift_team_only"):
        for i, p in enumerate(PROV + ["mean"]):
            v = [by_prov(rows, lambda r: r[nm][k])[i] for k in ks]
            print(f"{nm[:10]:10s}{p[:6]:6s}" + " ".join(f"{x:6.3f}" for x in v))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        report(t)
