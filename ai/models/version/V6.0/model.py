"""V6.0 의 신경망 정의 — 
GridEncoder 에 전역 요약 브랜치(global_fc + mix)를 추가.
conv 4층의 수용 영역은 9x9 라 9x18 보드의 가로 절반밖에 못 본다.
글로벌 평균 풀링으로 판 전체를 요약해 모든 셀에 붙여 준 뒤 1x1 conv 로 섞는다.
정책(RectangleMaskablePolicy)은 V5.0 과 동일하다.
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


class RectangleHead(nn.Module):
    """
    셀별 임베딩에서 (top_left 셀, bottom_right 셀) 쌍의 내적으로
    각 직사각형 행동의 로짓을 계산한다.

    핵심: 모든 위치가 같은 proj_tl / proj_br 파라미터를 공유하므로
    "합 10 패턴"을 위치마다 독립적으로 다시 배울 필요가 없다.
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
