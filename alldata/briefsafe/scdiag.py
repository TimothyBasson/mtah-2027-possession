"""Where a run gains or loses per provider: spell counts, precision / recall, team accuracy.

    python alldata/briefsafe/scdiag.py --tags pw1e pcr1e [--prov skillcorner]
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "learned"))
import score as S  # noqa: E402
from score import pm, decode  # noqa: E402


def one(args):
    name, f, cost = args
    raw, fold, spans = S.truths(name)
    z = np.load(f)
    out = []
    for t, dec, pre in ((raw, S.RAW, ""), (fold, S.FOLDED, "f")):
        p = pm.label_codes(decode.labels_from([z[f"{pre}s{i}"] for i in range(len(spans))], spans, len(t), **{**dec, "cost": dec["cost"] * cost}))
        r = pm.evaluate(t, p, bootstrap_n=0)
        h = r["spell_f1_at_headline"]
        sp = pm.spell_scores(t, pm.fill_abstentions(p, pm.valid_runs(t)), pm.valid_runs(t))["per_iou"][r["spell_iou_headline"]]
        out.append([r["spell_f1_mean"], r["n_gt_spells"], r["n_pred_spells"], sp.get("precision", np.nan), sp.get("recall", np.nan),
                    r["team_acc_gt_alive"], r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"]])
    return name, out


ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+"); ap.add_argument("--cost", type=float, default=1.0, help="decode switch cost x this"); ap.add_argument("--prov", nargs="*", default=["idsse", "metrica", "skillcorner"])
a = ap.parse_args()
with ProcessPoolExecutor(2) as ex:
    for pv in a.prov:
        for vi, view in enumerate(("built", "folded")):
            print(f"\n{pv} {view}:   spellF1   n_gt  n_pred  prec@h  rec@h  teamacc  pred_dur  gt_dur")
            for tag in a.tags:
                jobs = [(os.path.basename(f)[:-4], f, a.cost) for f in sorted(glob.glob(os.path.join(S.tag_dir(tag), pv, f"{pv}_*.npz")))]
                R = np.array([r[vi] for _, r in ex.map(one, jobs)], float)
                print(f"  {tag:10s}" + " ".join(f"{x:8.3f}" for x in np.nanmean(R, 0)))
