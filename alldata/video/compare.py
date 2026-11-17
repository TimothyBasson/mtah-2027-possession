"""Label-free comparison of video-ball files with the submission on BAR-ATH: frames that change,
dead share, possessions, and how the changes sit relative to the filled frames."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "possession")]
import load

RES = os.path.join(ROOT, "evaluation", "results")


def spells(l):
    k = np.isin(l, ["H", "A"])
    return int(np.sum(k[1:] & ((l[1:] != l[:-1]))) + k[0])


def main(ref="alldata_learned_tim_touch.csv", files=()):
    _, ball, _, halves = load.load_raw()
    g = np.zeros(len(ball), bool)
    for a, b in halves:
        g[a:b + 1] = True
    gap = g & np.isnan(ball[:, 0])
    r = pd.read_csv(os.path.join(RES, ref)).label.to_numpy().astype(str)
    print(f"{ref}: dead {np.mean(r[g]=='D'):.3f} (in tracker gaps {np.mean(r[gap]=='D'):.3f}), team spells {spells(r[g])}")
    for f in files:
        x = pd.read_csv(os.path.join(RES, f)).label.to_numpy().astype(str)
        ch = g & (x != r)
        print(f"{f}: changed {ch.sum()} frames ({ch.sum()/g.sum():.3%}; {np.mean(gap[ch]) if ch.any() else 0:.2f} of them in tracker gaps), "
              f"dead {np.mean(x[g]=='D'):.3f} (in gaps {np.mean(x[gap]=='D'):.3f}), team spells {spells(x[g])}; "
              f"changes D->team {np.sum(ch & (r=='D'))}, team->D {np.sum(ch & (x=='D'))}, team swap {np.sum(ch & (r!='D') & (x!='D'))}")


if __name__ == "__main__":
    main(files=sys.argv[1:])
