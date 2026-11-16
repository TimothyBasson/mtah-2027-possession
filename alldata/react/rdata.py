"""Loading for the team-reaction work: per held-out match, the base model's probabilities
(pw1e, the two heads averaged as run.py decodes them), the truths, and the reaction rows."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import react as R  # noqa: E402
from common import pm, decode  # noqa: E402

DECODE = dict(cost=4, min_team=1, min_dead=1)  # the submission's


class Match:
    def __init__(self, fold, name, path, key="s"):
        self.fold, self.name = fold, name
        self.probs, self.spans, self.raw, self.folded = C.load(path, key)
        z = np.load(os.path.join(R.OUT, name + ".npz"))
        self.R = [z[f"R{i}"] for i in range(len(self.spans))]
        self.fr = [z[f"fr{i}"] for i in range(len(self.spans))]
        for p, r in zip(self.probs, self.R):
            assert len(p) == len(r), (name, len(p), len(r))
        self.runs = {id(self.raw): pm.valid_runs(self.raw), id(self.folded): pm.valid_runs(self.folded)}

    def decode(self, probs=None, **kw):
        d = dict(DECODE, **kw)
        return pm.label_codes(decode.labels_from(self.probs if probs is None else probs, self.spans, len(self.raw), **d)).astype(np.int8)

    def score(self, pred):
        """spellF1 against (as built, folded)."""
        out = []
        for t in (self.raw, self.folded):
            runs = self.runs[id(t)]
            out.append(C.spell_f1(t, pm.fill_abstentions(pred, runs), runs))
        return out


def load_all(tag="pw1e"):
    return [Match(*m) for m in C.matches(tag)]


def brief_slice(m, pred, truth=None, max_s=5.0, fps=25):
    """Brief true possessions and what became of them: rows (start, stop, team, found at IoU 0.5,
    given to the other team = the other team holds most of its frames in the prediction)."""
    t = m.folded if truth is None else truth
    runs = m.runs[id(t)]
    _, g, p, fl = C.spell_f1(t, pm.fill_abstentions(pred, runs), runs, detail=True)
    rows = []
    for i, (a, b, k) in enumerate(g):
        if b - a >= max_s * fps:
            continue
        other = (pred[a:b] == 1 - k).mean() > 0.5
        rows.append((a, b, k, bool(fl[2, i]), bool(other and not fl[2, i])))
    return np.array(rows, np.int64).reshape(-1, 5)
