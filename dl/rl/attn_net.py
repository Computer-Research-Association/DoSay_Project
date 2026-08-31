"""
② 어텐션 정책망 (신경 조합최적화 스타일).
- 셀 162개를 토큰으로: 값 임베딩 + 위치 임베딩 → Transformer encoder → 셀 표현.
- feature map (B, d, 9, 18)로 reshape → CNN판과 동일 인터페이스(value, move_logits).
- pg.py 재사용 가능 (features/value/move_logits/infer 시그니처 동일).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 9, 18
N = ROWS * COLS


class AttnNet(nn.Module):
    def __init__(self, d=64, heads=4, layers=3, ff=128):
        super().__init__()
        self.val_emb = nn.Embedding(10, d)                 # 값 0-9
        self.pos_emb = nn.Parameter(torch.zeros(1, N, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.d = d
        self.v_fc = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
        self.p_fc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def features(self, x):
        """x: (B,10,9,18) one-hot → feature map (B, d, 9, 18)."""
        ids = x.argmax(dim=1).view(x.size(0), N)           # (B,162) 값 id
        h = self.val_emb(ids) + self.pos_emb               # (B,162,d)
        h = self.enc(h)                                    # (B,162,d)
        return h.transpose(1, 2).reshape(x.size(0), self.d, ROWS, COLS)

    def value(self, feat):
        z = feat.mean(dim=(2, 3))                          # (B,d)
        return torch.sigmoid(self.v_fc(z)).squeeze(-1)

    def move_logits(self, feat_single, moves):
        vecs = [feat_single[:, r1:r2 + 1, c1:c2 + 1].mean(dim=(1, 2))
                for (r1, c1, r2, c2) in moves]
        return self.p_fc(torch.stack(vecs, 0)).squeeze(-1)

    @torch.no_grad()
    def infer(self, grid, moves, device):
        from dl.az.net import encode
        x = torch.from_numpy(encode(grid)).unsqueeze(0).to(device)
        feat = self.features(x)
        v = float(self.value(feat)[0])
        if not moves:
            return np.zeros(0, dtype=np.float32), v
        p = torch.softmax(self.move_logits(feat[0], moves), 0).cpu().numpy()
        return p, v
