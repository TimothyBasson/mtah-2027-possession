"""Joint feature cache at every 5th frame: control features + seq features, both orientations."""
from lib import *
from seq import model as sm
from control_features import match_ball
J = ROOT + "/data/cache/wip/joint"; os.makedirs(J, exist_ok=True)
def one(n):
    if os.path.exists(f"{J}/{n}.npz"): return n
    d = dict(np.load(f"{ROOT}/data/external/{n}.npz")); ball = match_ball(n, d)
    Xq, nq, fr = sm.featurise(d, ball); Xqs = sm.featurise(d, ball, swap=True)[0]
    z = np.load(f"{ROOT}/data/cache/control_{n}.npz"); nc = ["c_" + s for s in z["names"]]
    X = np.hstack([z["X"][fr], Xq]); Xs = np.hstack([z["Xs"][fr], Xqs])
    m = load(n)
    np.savez(f"{J}/{n}.npz", X=X, Xs=Xs, names=np.array(nc + list(nq)), frames=fr, y=m["folded"][fr], yraw=m["labels"][fr])
    return n
if __name__ == "__main__":
    with Pool(3) as p:
        for n in p.imap_unordered(one, NAMES): print(n, flush=True)
