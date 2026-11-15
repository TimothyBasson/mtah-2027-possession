"""Coordinate descent over decoder settings; objective = mean folded spellF1 over the given sets.
Memo of every evaluated config in logs_tune/memo.pkl (all three sets scored each time)."""
import json, os, pickle, sys
import tl
MEMO = os.path.join(tl.ROOT, "logs_tune", "memo.pkl")
memo = pickle.load(open(MEMO, "rb")) if os.path.exists(MEMO) else {}
GRID = dict(weight=[0.1, 0.125, 0.15, 0.175, 0.2, 0.25], cw=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            width_weight=[0.0, 0.025, 0.05, 0.075], poss=[8, 12, 16, 20, 30, 45], stop=[4, 8, 16],
            to_dead=[0.3, 0.4, 0.5, 0.6], smooth=[0, 5, 12, 18], w_gap=[0.0, 0.5, 0.75, 1.0, 1.25],
            w_place=[0.5, 1.0, 1.5, 2.0], w_speed=[0.0, 0.5, 1.0], w_flight=[0.5, 0.75, 1.0, 1.25],
            tabs=[None, ("folded", "pair")], cw_dead=[None, 0.1, 0.3, 0.5])
def key(c): return json.dumps(c, sort_keys=True)
def evaluate(cfgs):
    todo = [c for c in cfgs if key(c) not in memo]
    for c, o in zip(todo, tl.eval_configs(todo)):
        memo[key(c)] = {k: o[k] for k in tl.SETS}
    if todo: pickle.dump(memo, open(MEMO, "wb"))
    return [memo[key(c)] for c in cfgs]
def descend(sets, base, passes=3):
    obj = lambda o: sum(o[k] for k in sets) / len(sets)
    cur = dict(base); best = obj(evaluate([cur])[0])
    for p in range(passes):
        moved = False
        for name, vals in GRID.items():
            cands = [dict(cur, **{name: v}) for v in vals]
            scores = [obj(o) for o in evaluate(cands)]
            i = max(range(len(vals)), key=lambda j: scores[j])
            if scores[i] > best + 1e-4:
                cur, best, moved = cands[i], scores[i], True
                print("  pass %d %s -> %s  %.4f" % (p, name, vals[i], best), flush=True)
        if not moved: break
    return cur, best
if __name__ == "__main__":
    base = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    res = {}
    for sets in (tl.SETS, ("metrica", "skillcorner"), ("idsse", "skillcorner"), ("idsse", "metrica")):
        print("select on", sets, flush=True)
        cur, best = descend(sets, base)
        o = evaluate([cur])[0]
        print("  ->", cur, "| %.3f / %.3f / %.3f mean %.4f" % (o["idsse"], o["metrica"], o["skillcorner"], sum(o.values()) / 3), flush=True)
        res[sets] = o
    fair = [res[s][k] for s, k in ((("metrica", "skillcorner"), "idsse"), (("idsse", "skillcorner"), "metrica"), (("idsse", "metrica"), "skillcorner"))]
    print("fair (each set scored with settings picked on the other two): %.3f / %.3f / %.3f mean %.4f" % (*fair, sum(fair) / 3))
