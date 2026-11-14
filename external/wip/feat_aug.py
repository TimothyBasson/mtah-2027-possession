"""Joint features for the broadcast-like copies of IDSSE and Metrica (external/broadcast.py).

    python external/broadcast.py --tag bc         # control features of the copies
    python external/wip/feat_aug.py bc            # -> data/cache/wip/joint/bc+<name>.npz

Same columns as feat.py; the labels are the original match's."""
from lib import *
import ast
import broadcast as bc
from seq import model as sm
from score import drop_ball
from control_features import SEEN
J = ROOT + "/data/cache/wip/joint"
TAG = sys.argv[1] if len(sys.argv) > 1 else "bc"
def one(n):
    out = f"{J}/{TAG}+{n}.npz"
    if os.path.exists(out): return n
    d = dict(np.load(f"{ROOT}/data/external/{n}.npz"))
    z = np.load(bc.cache_path(TAG, n))
    pxy, bxy = bc.broadcast_like(d, n, **ast.literal_eval(str(z["options"])))
    ball = np.column_stack([drop_ball(bxy, SEEN, int(d["fps"])), np.full(len(bxy), np.nan, np.float32)])
    d["players"] = np.dstack([pxy, d["players"][..., 2]])
    Xq, nq, fr = sm.featurise(d, ball); Xqs = sm.featurise(d, ball, swap=True)[0]
    nc = ["c_" + s for s in z["names"]]
    m = load(n)
    np.savez(out, X=np.hstack([z["X"][fr], Xq]), Xs=np.hstack([z["Xs"][fr], Xqs]), names=np.array(nc + list(nq)),
             frames=fr, y=m["folded"][fr], yraw=m["labels"][fr])
    return n
if __name__ == "__main__":
    with Pool(3) as p:
        for n in p.imap_unordered(one, [n for n in NAMES if n.startswith(bc.SOURCES)]): print(n, flush=True)
