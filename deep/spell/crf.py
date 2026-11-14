"""A chain over H / A / D in which a run must last a minimum time, trainable through its decode.

Each class is a short ladder of sub-states: a run enters at the bottom, must climb one rung per
step, and may only leave from the top, where it can also stay. So no path, in training or at
prediction, holds a run shorter than the ladder (an explicit minimum-duration HMM; Rabiner 1989;
the conditional form is the semi-Markov CRF of Sarawagi & Cohen 2004 restricted to a duration
floor). The cost of each kind of change (H->A, H->D, ...) is learned.

Training is on whole sequences, not frames: loss = log Z(all paths) - log Z(paths that agree with
the labels), the CRF likelihood (Lafferty et al. 2001). With a margin, wrong frames are charged in
log Z(all paths) (softmax-margin, Gimpel & Smith 2010), and the charge per frame can be one over
its possession's length, so that every possession weighs the same, as it does in spellF1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


class MinDurCRF(nn.Module):
    def __init__(self, mins=(10, 10, 5), init_cost=3.0):
        super().__init__()
        self.mins = tuple(int(m) for m in mins)
        cls, top, bottom = [], [], []
        for k, m in enumerate(self.mins):
            bottom.append(len(cls))
            cls += [k] * m
            top.append(len(cls) - 1)
        S = len(cls)
        self.register_buffer("cls", torch.tensor(cls))
        fixed = torch.full((S, S), NEG)
        for k, m in enumerate(self.mins):
            for j in range(m - 1):
                fixed[bottom[k] + j, bottom[k] + j + 1] = 0
            fixed[top[k], top[k]] = 0
        self.register_buffer("fixed", fixed)
        start = torch.full((S,), NEG)  # a sequence opens mid-run: at the top of a ladder, free to leave
        start[top] = 0
        self.register_buffer("start", start)
        idx = [(top[k], bottom[q], k, q) for k in range(3) for q in range(3) if k != q]
        self.register_buffer("sw_from", torch.tensor([i[0] for i in idx]))
        self.register_buffer("sw_to", torch.tensor([i[1] for i in idx]))
        self.register_buffer("sw_kq", torch.tensor([i[2] * 3 + i[3] for i in idx]))
        self.raw_cost = nn.Parameter(torch.full((9,), float(np.log(np.expm1(init_cost)))))

    def costs(self):
        return F.softplus(self.raw_cost).reshape(3, 3)

    def trans(self):
        A = self.fixed.clone()
        A[self.sw_from, self.sw_to] = -F.softplus(self.raw_cost)[self.sw_kq]
        return A

    def expand(self, E, y=None):
        """(B, T, 3) class scores -> (B, T, S); with y, states of another class are shut."""
        Ee = E[..., self.cls]
        if y is not None:
            ok = (y[..., None] == self.cls) | (y[..., None] < 0)
            Ee = Ee.masked_fill(~ok, NEG * 10)
        return Ee

    def log_z(self, Ee):
        A = self.trans()
        alpha = Ee[:, 0] + self.start
        for t in range(1, Ee.shape[1]):
            alpha = torch.logsumexp(alpha[:, :, None] + A, 1) + Ee[:, t]
        return torch.logsumexp(alpha, -1)

    def nll(self, E, y, cost=None):
        """Per-sequence loss. cost (B, T): what a wrong class at that frame is charged."""
        free = E
        if cost is not None:
            wrong = F.one_hot(y.clamp(min=0), 3) == 0
            free = E + cost[..., None] * wrong * (y >= 0)[..., None]
        z = self.log_z(torch.cat([self.expand(free), self.expand(E, y)]))  # both in one pass
        return z[:len(E)] - z[len(E):]

    @torch.no_grad()
    def viterbi(self, E, lengths=None):
        """(B, T, 3) -> (B, T) classes on the best path. lengths: valid steps per sequence."""
        Ee, A = self.expand(E), self.trans()
        B, T, S = Ee.shape
        back = torch.zeros(B, T, S, dtype=torch.uint8)
        v = Ee[:, 0] + self.start
        ar = torch.arange(B)
        if lengths is None:
            lengths = torch.full((B,), T)
        for t in range(1, T):
            sc, bp = (v[:, :, None] + A).max(1)
            live = (t < lengths)[:, None]
            v = torch.where(live, sc + Ee[:, t], v)
            back[:, t] = torch.where(live, bp, torch.arange(S)[None]).to(torch.uint8)
        path = torch.zeros(B, T, dtype=torch.long)
        s = v.argmax(-1)
        for t in range(T - 1, -1, -1):
            path[:, t] = s
            s = back[ar, t, s].long()
        return self.cls[path]

    @torch.no_grad()
    def sample(self, E, n):
        """n paths per sequence drawn from the chain's distribution: (B, n, T) sub-states."""
        Ee, A = self.expand(E), self.trans()
        B, T, S = Ee.shape
        alphas = [Ee[:, 0] + self.start]
        for t in range(1, T):
            alphas.append(torch.logsumexp(alphas[-1][:, :, None] + A, 1) + Ee[:, t])
        out = torch.zeros(B, n, T, dtype=torch.long)
        s = torch.multinomial(F.softmax(alphas[-1], -1), n, replacement=True)
        out[:, :, -1] = s
        for t in range(T - 2, -1, -1):
            lp = alphas[t][:, None, :] + A.T[s]  # (B, n, S): into the state already drawn
            s = torch.multinomial(F.softmax(lp, -1).reshape(B * n, S), 1).reshape(B, n)
            out[:, :, t] = s
        return out

    def path_score(self, E, states):
        """Unnormalised score of sub-state paths (B, n, T), differentiable in E and the costs."""
        Ee, A = self.expand(E), self.trans()
        em = torch.gather(Ee[:, None].expand(-1, states.shape[1], -1, -1), 3, states[..., None])[..., 0].sum(-1)
        return em + A[states[:, :, :-1], states[:, :, 1:]].sum(-1) + self.start[states[:, :, 0]]
