import numpy as np, tl, sys
import possession_metrics as pm
c = dict(tl.DEFAULT)
for k in tl.SETS:
    acc = []
    for n in [n for n in tl.names() if n.startswith(k)][:4]:
        m = tl.match(n); pred = tl.decode(n, c)
        g, p = pm.label_codes(m["folded"]), pm.label_codes(pred)
        runs = pm.valid_runs(g); p = pm.fill_abstentions(p, runs)
        s = pm.spell_scores(g, p, runs)
        gr = (m["folded"] != "X")
        gt, pr = m["folded"][gr], pred[gr]
        alive = (gt != "D") & (pr != "D")
        acc.append([s["n_gt"], s["n_pred"]] + [s["per_iou"][t][q] for t in sorted(s["per_iou"]) for q in ("precision", "recall")]
                   + [np.mean(gt == "D"), np.mean(pr == "D"), np.mean(gt[alive] == pr[alive]),
                      np.mean((gt == "D") & (pr != "D")), np.mean((gt != "D") & (pr == "D"))])
    print(k, sorted(s["per_iou"]))
    print("  n_gt n_pred | P,R per IoU | D share gt, pred | team acc | dead->live, live->dead")
    print("  ", np.round(np.mean(acc, axis=0), 3))
