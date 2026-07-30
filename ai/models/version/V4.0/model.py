"""V4.0 의 신경망 정의.
관측(0~9 정수)을 0~1 로 정규화해 conv 3층에 통과시킨 뒤 Flatten + Linear 로 256차원 특징 하나로 압축
정책은 sb3_contrib 기본 MaskableActorCriticPolicy 를 쓰므로, 출력층은 SB3가 자동으로 붙이는 Linear(256 -> 7533)
"""

import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

POLICY_CLASS = MaskableActorCriticPolicy


class GridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        rows, cols = observation_space.shape # type: ignore

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, 1, rows, cols)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (N, rows, cols), 값 범위 0~9
        x = observations.float().unsqueeze(1) / 9.0  # (N, 1, rows, cols), 0~1 정규화
        return self.linear(self.cnn(x))


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        normalize_images=False,
    )
