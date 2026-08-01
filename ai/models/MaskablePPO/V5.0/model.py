"""V5.0 의 신경망 정의.
1. GridEncoder: 값을 one-hot 10채널로 넣고, Flatten 후 Linear 병목을 없애 셀별 임베딩(embed_dim x rows x cols)을 그대로 정책에 넘긴다.
2. RectangleHead: 행동을 (좌상단, 우하단) 셀 쌍으로 보고 두 임베딩의 내적으로 로짓을 만든다. 
                  모든 위치가 proj_tl / proj_br 을 공유하므로 위치마다 같은 패턴을 다시 학습할 필요가 없다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ai.envs.action_sets import get_all_action
from game.action import Action


class GridEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim: int = 64):
        rows, cols = observation_space.shape  # type: ignore
        super().__init__(observation_space, features_dim=embed_dim * rows * cols)
        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (N, rows, cols), 값 0~9
        x = F.one_hot(observations.long(), num_classes=10)  # (N, R, C, 10)
        x = x.permute(0, 3, 1, 2).float()                   # (N, 10, R, C)
        return self.cnn(x).flatten(1)                       # (N, embed_dim*R*C)


class RectangleHead(nn.Module):
    """
    셀별 임베딩에서 (top_left 셀, bottom_right 셀) 쌍의 내적으로
    각 직사각형 행동의 로짓을 계산한다.

    핵심: 모든 위치가 같은 proj_tl / proj_br 파라미터를 공유하므로
    "합 10 패턴"을 위치마다 독립적으로 다시 배울 필요가 없다.
    (기존 Linear(256 -> 7533) 출력층이 갖던 문제를 출력단에서 해결)
    """

    def __init__(self, rows: int, cols: int, embed_dim: int,
                 actions: list[Action], head_dim: int = 32):
        super().__init__()
        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim
        self.proj_tl = nn.Linear(embed_dim, head_dim)
        self.proj_br = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5

        # 행동 인덱스 -> (tl 셀 인덱스, br 셀 인덱스) 매핑을 버퍼로 고정
        tl_idx = torch.tensor(
            [a.top_left[0] * cols + a.top_left[1] for a in actions], dtype=torch.long)
        br_idx = torch.tensor(
            [a.bottom_right[0] * cols + a.bottom_right[1] for a in actions], dtype=torch.long)
        self.register_buffer("tl_idx", tl_idx)
        self.register_buffer("br_idx", br_idx)

        # 초기 로짓을 0 근처로 -> 초기 정책이 (마스크 내) 균등 분포에 가깝게
        for m in (self.proj_tl, self.proj_br):
            nn.init.orthogonal_(m.weight, gain=0.3)
            nn.init.zeros_(m.bias)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        n = latent_pi.shape[0]
        # (N, E*R*C) -> (N, R*C, E)
        cells = latent_pi.view(n, self.embed_dim, self.rows * self.cols).transpose(1, 2)
        tl = self.proj_tl(cells)                              # (N, RC, d)
        br = self.proj_br(cells)                              # (N, RC, d)
        pair = torch.bmm(tl, br.transpose(1, 2)) * self.scale  # (N, RC, RC)
        return pair[:, self.tl_idx, self.br_idx]              # (N, n_actions)


class RectangleMaskablePolicy(MaskableActorCriticPolicy):
    """
    MaskableActorCriticPolicy에서 action_net(Linear)만 RectangleHead로 교체.
    사용 조건:
      - features_extractor 는 GridEncoder (rows/cols/embed_dim 속성 필요)
      - net_arch=dict(pi=[], vf=[...])  # pi를 비워 latent_pi = 셀별 임베딩 그대로
    """

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = 32, **kwargs):
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)를 넘겨주세요."
        self._actions = actions
        self._head_dim = head_dim
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        fe = self.features_extractor  # GridEncoder
        assert self.mlp_extractor.latent_dim_pi == fe.features_dim, \
            "net_arch의 pi는 반드시 빈 리스트([])여야 합니다."

        self.action_net = RectangleHead(
            fe.rows, fe.cols, fe.embed_dim, self._actions, self._head_dim # type: ignore
        )
        # action_net을 교체했으므로 옵티마이저 재생성 (새 파라미터 포함, 기존 Linear 제외)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs # type: ignore
        )


POLICY_CLASS = RectangleMaskablePolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(embed_dim=64),
        net_arch=dict(pi=[], vf=[256, 256]),  # pi=[] 필수: latent_pi = 셀별 임베딩 그대로
        normalize_images=False,
        actions=get_all_action(rows, cols),   # RectangleMaskablePolicy 전용 인자
        head_dim=32,
    )
