"""The network: a set encoder over the players of each frame, then a temporal model over frames.

After Kim et al. (KDD 2023, "Ball trajectory inference from multi-agent sports contexts using set
transformer and hierarchical Bi-LSTM") and TranSPORTmer (Capellera et al. 2024): players are an
unordered set, so each frame is encoded with attention that ignores slot order, which is what our
tracker's identity-free, swapping slots need. Frames are then joined in time by dilated
convolutions and a two-way GRU. Heads: H / A / D per frame, and where the ball is (training
target only; inferring the ball from the players is the auxiliary task both papers use).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, pad):
        h = self.n1(x)
        x = x + self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class FrameEncoder(nn.Module):
    """(N, K, F) players + (N, K) mask -> (N, out). Pools each team apart, home first."""

    def __init__(self, nfeat, d=64, heads=4, blocks=2, out=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nfeat, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(SetBlock(d, heads) for _ in range(blocks))
        self.out = nn.Sequential(nn.Linear(4 * d + 2, out), nn.GELU())

    def forward(self, x, mask):
        team = x[..., 6]
        empty = ~mask.any(1)
        pad = ~mask
        pad = pad & ~empty[:, None]  # a frame with nobody tracked must not be all-padding
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, pad)
        pooled = []
        for sign in (1, -1):
            m = (mask & (team * sign > 0)).unsqueeze(-1)
            cnt = m.sum(1).clamp(min=1)
            mean = (h * m).sum(1) / cnt
            mx = h.masked_fill(~m, -1e4).max(1)[0] * (m.sum(1) > 0)
            pooled += [mean, mx, m.sum(1).float() / 11]
        return self.out(torch.cat(pooled, -1)), h


class TCNBlock(nn.Module):
    def __init__(self, d, dilation, drop):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(d, d, 1)
        self.norm = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.c2(F.gelu(self.c1(self.norm(x)))))


class PossessionNet(nn.Module):
    """ptr=False is v2. ptr=True (v4) adds the ball-taught middle stage: after the convolutions the
    net points at the player it believes is on the ball (a distribution over the frame's players,
    taught by the real ball on the training matches), and what it pointed at - that player's
    features and how the pointing splits between the teams - is handed to the GRU. The ball itself
    is never an input: forward() takes the players and their mask and nothing else."""

    def __init__(self, nfeat=8, d_set=64, d=128, dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32),
                 gru=True, drop=0.1, set_blocks=2, ptr=False):
        super().__init__()
        self.enc = FrameEncoder(nfeat, d_set, blocks=set_blocks, out=d)
        self.tcn = nn.Sequential(*[TCNBlock(d, k, drop) for k in dilations])
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True) if gru else None
        self.cls = nn.Linear(d, 3)
        self.ball = nn.Linear(d, 2)
        self.edge = nn.Linear(d, 1)  # "possession changes hands or play stops/starts within 0.4 s"
        self.ptr = ptr
        if ptr:
            self.ptr_q = nn.Linear(d, d_set)
            self.ptr_k = nn.Linear(d_set, d_set)
            self.ptr_mix = nn.Sequential(nn.Linear(d + d_set + 4, d), nn.GELU(), nn.Linear(d, d))
            self.bvel = nn.Linear(d, 2)    # the ball's velocity
            self.touch = nn.Linear(d, 1)   # "the ball is struck or stopped within 0.2 s"
            self.free = nn.Linear(d, 1)    # how far the ball is from the nearest player (log metres)

    def features(self, x, mask):
        """(B, T, d) per-frame features after the last stage, and the ball-taught stage's outputs."""
        B, T, K, Fd = x.shape
        h, e = self.enc(x.reshape(B * T, K, Fd), mask.reshape(B * T, K))
        h = h.reshape(B, T, -1)
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        extra = {}
        if self.ptr:
            e = e.reshape(B, T, K, -1)
            s = (self.ptr_q(h).unsqueeze(2) * self.ptr_k(e)).sum(-1) / e.shape[-1] ** 0.5
            s = s.masked_fill(~mask, -1e4)
            p = torch.softmax(s, -1) * mask.any(-1, keepdim=True)
            team = x[..., 6]
            mass = torch.stack([(p * (team > 0)).sum(-1), (p * (team < 0)).sum(-1)], -1)
            at = (p.unsqueeze(-1) * x[..., :2]).sum(2)  # where the pointed-at players stand
            h = h + self.ptr_mix(torch.cat([h, (p.unsqueeze(-1) * e).sum(2), mass, at], -1))
            extra["ptr"] = s
        if self.gru is not None:
            h = h + self.gru(h)[0]
        return (h, extra) if self.ptr else h

    def forward(self, x, mask):
        extra = {}
        h = self.features(x, mask)
        if self.ptr:
            h, extra = h
            extra.update(bvel=self.bvel(h), touch=self.touch(h).squeeze(-1), free=self.free(h).squeeze(-1))
        return self.cls(h), self.ball(h), self.edge(h).squeeze(-1), extra


def load_net(path, nfeat=8):
    """A saved net of either kind; which kind is read off its weights."""
    sd = torch.load(path, map_location="cpu")
    net = PossessionNet(nfeat=nfeat, ptr="ptr_q.weight" in sd)
    net.load_state_dict(sd)
    return net.eval()
