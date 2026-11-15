"""The turnover detector: each player's path in time and the players of a frame as a set, in turn.

Blocks alternate a convolution along each player's own track (what he does over about a second)
with attention across the players of a frame (who is near whom). Then the net points at the
player on the ball (taught by the real ball on training matches, as deep/'s bt) and says, per
player, whether he touches it now. Frame heads read the pooled teams and what was pointed at:
"the ball passes to home / to away at this frame". forward() takes the players and nothing else.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, dilation, heads=4):
        super().__init__()
        self.nt = nn.GroupNorm(8, d)
        self.ct = nn.Conv1d(d, d, 5, padding=2 * dilation, dilation=dilation)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, mask):
        B, T, K, d = h.shape
        m = mask.unsqueeze(-1)
        x = h.permute(0, 2, 3, 1).reshape(B * K, d, T)
        x = self.ct(F.gelu(self.nt(x))).reshape(B, K, d, T).permute(0, 3, 1, 2)
        h = (h + x) * m
        x = h.reshape(B * T, K, d)
        pad = ~mask.reshape(B * T, K)
        pad = pad & pad.all(1, keepdim=True).logical_not()  # an empty frame must not be all padding
        q = self.n1(x)
        x = x + self.att(q, q, q, key_padding_mask=pad, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x.reshape(B, T, K, d) * m


class TCN(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.n = nn.GroupNorm(8, d)
        self.c1 = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(d, d, 1)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(self.n(x))))


class TurnNet(nn.Module):
    def __init__(self, nfeat=9, d=64, dilations=(1, 2, 4), frame_d=128, tcn=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nfeat, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, k) for k in dilations)
        self.ptr = nn.Linear(d, 1)
        self.touch = nn.Linear(d, 1)
        self.join = nn.Sequential(nn.Linear(5 * d + 6, frame_d), nn.GELU())
        self.tcn = nn.Sequential(*[TCN(frame_d, k) for k in tcn])
        self.ev = nn.Linear(frame_d, 3)    # nothing / to home / to away, labels as given
        self.evf = nn.Linear(frame_d, 3)   # the same on labels folded to the control rule
        self.cls = nn.Linear(frame_d, 3)   # H / A / D, a side task

    def forward(self, x, mask):
        team = x[..., 7]
        h = self.inp(x) * mask.unsqueeze(-1)
        for b in self.blocks:
            h = b(h, mask)
        s = self.ptr(h).squeeze(-1).masked_fill(~mask, -1e4)
        p = torch.softmax(s, -1) * mask.any(-1, keepdim=True)
        touch = self.touch(h).squeeze(-1)
        parts = []
        for sign in (1, -1):
            m = (mask & (team * sign > 0)).unsqueeze(-1)
            cnt = m.sum(2).clamp(min=1)
            parts += [(h * m).sum(2) / cnt, h.masked_fill(~m, -1e4).max(2)[0] * (m.sum(2) > 0), m.sum(2).float() / 11,
                      (p * (team * sign > 0)).sum(-1, keepdim=True),
                      (torch.sigmoid(touch) * (mask & (team * sign > 0))).sum(-1, keepdim=True).clamp(max=1)]
        parts.append((p.unsqueeze(-1) * h).sum(2))
        f = self.join(torch.cat(parts, -1))
        f = self.tcn(f.transpose(1, 2)).transpose(1, 2)
        return dict(ev=self.ev(f), evf=self.evf(f), cls=self.cls(f), ptr=s, touch=touch)


def load_net(path):
    net = TurnNet()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()
