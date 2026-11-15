import numpy as np, tl
c = dict(tl.DEFAULT)
def trans(lab):
    i = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return [(int(f), lab[f - 1], lab[f]) for f in i if lab[f] != "X" and lab[f - 1] != "X"]
for k in tl.SETS:
    offs = {"TT": [], "TD": [], "DT": []}
    for n in [n for n in tl.names() if n.startswith(k)][:5]:
        m = tl.match(n); pred = tl.decode(n, c)
        pt = trans(pred)
        for f, a, b in trans(m["folded"]):
            typ = "TD" if b == "D" else "DT" if a == "D" else "TT"
            cand = [pf - f for pf, pa, pb in pt if pb == b and (typ != "TT" or pa == a) and (pa == "D") == (a == "D") and abs(pf - f) <= 75]
            if cand:
                offs[typ].append(min(cand, key=abs))
    print(k, {t: (len(v), "median %+.0f mean %+.1f p25 %+.0f p75 %+.0f" % (np.median(v), np.mean(v), np.percentile(v, 25), np.percentile(v, 75))) for t, v in offs.items()})
