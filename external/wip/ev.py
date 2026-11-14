from lib import *
import sys
FILES = sys.argv[1:]
WS = (.05, .1, .15, .2, .3)
def one(args):
    f, n = args; L = loso(f); m = load(n); r = ratio(L, n)
    return [sf1(m["folded"], decode(m, w * r)) for w in WS]
if __name__ == "__main__":
    for f in FILES:
        names = [n for n in NAMES if n in np.load(f"{ROOT}/data/cache/{f}").files]
        with Pool(4) as p: R = p.map(one, [(f, n) for n in names])
        for i, w in enumerate(WS): report({n: R[j][i] for j, n in enumerate(names)}, f"{f[:-4]} w{w}")
