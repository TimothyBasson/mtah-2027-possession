from lib import *
L = loso()
def absorb(min_s, fps=25):
    def f(p):
        p = p.copy(); k = int(min_s*fps)
        while True:
            st = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]; en = np.r_[st[1:], len(p)]
            ch = False
            for i in range(1, len(st)-1):
                a, b = st[i], en[i]
                if p[a] in "HA" and b-a < k and p[a-1] == p[b] and p[a-1] != p[a] and p[a-1] in "HA":
                    p[a:b] = p[a-1]; ch = True
            if not ch: return p
    return f
CFG = {"v6": dict(w=0), "v7 w.15": dict(w=.15), "v7 w.2": dict(w=.2), "v7 w.3": dict(w=.3),"v7 w.5": dict(w=.5),
       "w.15 absorb1": dict(w=.15, post=absorb(1)), "w.15 absorb2": dict(w=.15, post=absorb(2)), "w.15 absorb3": dict(w=.15, post=absorb(3)),
       "w.15 poss20": dict(w=.15, tr=(20,8,.4)), "w.15 poss8": dict(w=.15, tr=(8,8,.4)), "w.15 stop5": dict(w=.15, tr=(12,5,.4)),"w.15 stop12": dict(w=.15, tr=(12,12,.4)),
       "w.15 dead.2": dict(w=.15, tr=(12,8,.2)), "w.15 dead.6": dict(w=.15, tr=(12,8,.6))}
def one(n):
    m = load(n); r = ratio(L, n); out = {}
    for k, c in CFG.items():
        tr = None
        if "tr" in c:
            from hmm.model import rate_log_trans
            tr = rate_log_trans(c["tr"][0]*25, c["tr"][1]*25, c["tr"][2])
        pred = decode(m, c["w"]*r if c["w"] else None, tr, c.get("post"))
        out[k] = sf1(m["folded"], pred)
    return out
if __name__ == "__main__":
    R = run_all(one)
    for k in CFG: report({n: R[n][k] for n in R}, k)
