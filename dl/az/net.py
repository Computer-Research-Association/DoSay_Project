"""
Policy + Value 망 (AlphaZero).
- 입력: (B, 10, 9, 18)  값 0-9 one-hot.
- 트렁크: conv 잔차 블록 → feature map (B, C, 9, 18).
- value head: 전역 pool → FC → [0,1] 스칼라 (도달가능점수/162).
- policy: 각 수(사각형)를 ROI 평균 pool → FC → 로짓 → 합법수 softmax.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 9, 18
CH = 64


def encode(grid):
    """(9,18) int8 → (10,9,18) float one-hot."""
    x = np.zeros((10, ROWS, COLS), dtype=np.float32)
    g = np.asarray(grid)
    for v in range(10):
        x[v] = (g == v)
    return x


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, ch=CH, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(10, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU())
        self.res = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # value head
        self.v_conv = nn.Sequential(nn.Conv2d(ch, 8, 1), nn.BatchNorm2d(8), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(8 * ROWS * COLS, 128), nn.ReLU(), nn.Linear(128, 1))
        # policy head (ROI 벡터 → 로짓)
        self.p_fc = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Linear(64, 1))

    def features(self, x):
        return self.res(self.stem(x))               # (B, ch, 9, 18)

    def value(self, feat):
        z = self.v_conv(feat).flatten(1)
        return torch.sigmoid(self.v_fc(z)).squeeze(-1)   # (B,) in [0,1]

    def move_logits(self, feat_single, moves):
        """feat_single: (ch,9,18), moves: [(r1,c1,r2,c2)] → (M,) 로짓."""
        vecs = []
        for (r1, c1, r2, c2) in moves:
            vecs.append(feat_single[:, r1:r2 + 1, c1:c2 + 1].mean(dim=(1, 2)))
        V = torch.stack(vecs, dim=0)                 # (M, ch)
        return self.p_fc(V).squeeze(-1)              # (M,)

    @torch.no_grad()
    def infer(self, grid, moves, device):
        """단일 상태 추론 → (policy 확률 np(M,), value float)."""
        x = torch.from_numpy(encode(grid)).unsqueeze(0).to(device)
        feat = self.features(x)
        v = float(self.value(feat)[0])
        if not moves:
            return np.zeros(0, dtype=np.float32), v
        logits = self.move_logits(feat[0], moves)
        p = torch.softmax(logits, dim=0).cpu().numpy()
        return p, v
