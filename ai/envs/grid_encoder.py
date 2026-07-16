# ai/envs/grid_encoder.py  (V6: global context 추가)
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim: int = 64):
        rows, cols = observation_space.shape  # type: ignore
        super().__init__(observation_space, features_dim=embed_dim * rows * cols)
        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.global_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(64 + 64, embed_dim, kernel_size=1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(observations.long(), num_classes=10)  # (N, R, C, 10)
        x = x.permute(0, 3, 1, 2).float()                   # (N, 10, R, C)

        h = self.cnn(x)                                     # (N, 64, R, C)  국소 패턴
        g = self.global_fc(h.mean(dim=(2, 3)))              # (N, 64)        전역 요약
        g = g[:, :, None, None].expand(-1, -1, self.rows, self.cols)

        cells = self.mix(torch.cat([h, g], dim=1))          # (N, E, R, C)
        return cells.flatten(1)                             # (N, E*R*C)