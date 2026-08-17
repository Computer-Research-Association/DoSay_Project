"""QR-DQN 신경망 — 행동마다 값 하나가 아니라 분포를 학습한다.

**왜 분포인가.** V8 계열의 실패는 '형제 상태를 서열 매기지 못하는 것' 이었다.
explained_variance 가 0.983 인데도 탐색이 +2점밖에 못 냈는데, 그 잔차(약 1.7점)가
형제 간 가치 차이와 같은 크기였기 때문이다. 값 하나로 압축하면 '평균은 같지만
하나는 안정적이고 하나는 도박' 인 두 수가 구분되지 않는다. 분위수를 학습하면
같은 파라미터로 회귀 목표가 1개에서 N개로 늘어 훨씬 촘촘한 신호를 받는다.

**분위수 개수.** sb3_contrib 기본값은 200 인데 여기서는 쓸 수 없다. 행동이
7533개라 (배치, 분위수, 행동) 텐서가 배치 256 x 분위수 200 이면 1.5TB 다.
16개로 줄이면 배치 128 에서 62MB 로 떨어진다. 분포 학습의 이득 대부분은
수십 개 분위수에서 이미 나오므로 이 교환은 남는 장사다.

**헤드 구조.** 분위수마다 꼭짓점 상호작용을 따로 계산하면 bmm 이 16배가 되어
또 터진다. 그래서 위치 항(모든 분위수 공통)과 분위수별 오프셋을 나눈다.

    q(a, k) = base(a)  +  spread_k(a)  +  level[k]
              ^공통       ^분위수별 내부 요약   ^분위수별 상수

base 는 DQN V1.0 과 같은 (꼭짓점 + 내부 요약 + 기하x사과수) 합이고,
spread 는 분위수 개수만큼의 평면을 누적합으로 훑어 얻는다. 둘 다 gather 네 번씩이면
7533개 행동 전부가 계산된다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.qrdqn.policies import QRDQNPolicy, QuantileNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ai.envs.action_sets import get_all_action
from game.action import Action

WIDTH = 64
N_BLOCKS = 4
EMBED_DIM = 48
HEAD_DIM = 32
N_GROUPS = 8
MAX_APPLE_COUNT = 10
TARGET_SUM = 10
MASK_FILL = -1e8
N_QUANTILES = 16     # 7533개 행동 때문에 200 대신 16. 위 주석 참고.


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    """(N, C, R, C) 또는 (N, R, C) -> 마지막 두 축에 0 패딩 누적합."""
    padded = F.pad(plane, (1, 0, 1, 0))
    return padded.cumsum(dim=-2).cumsum(dim=-1)


class RectIndex(nn.Module):
    """행동 -> 직사각형 모서리 좌표. 누적합에서 부분합을 뽑는 데 쓴다."""

    def __init__(self, actions: list[Action]):
        super().__init__()
        r1 = torch.tensor([a.top_left[0] for a in actions], dtype=torch.long)
        c1 = torch.tensor([a.top_left[1] for a in actions], dtype=torch.long)
        r2 = torch.tensor([a.bottom_right[0] for a in actions], dtype=torch.long)
        c2 = torch.tensor([a.bottom_right[1] for a in actions], dtype=torch.long)
        for name, value in (("r_lo", r1), ("c_lo", c1), ("r_hi", r2 + 1), ("c_hi", c2 + 1)):
            self.register_buffer(name, value)

    @staticmethod
    def _corners(prefix, r_lo, c_lo, r_hi, c_hi) -> torch.Tensor:
        return (prefix[..., r_hi, c_hi] - prefix[..., r_lo, c_hi]
                - prefix[..., r_hi, c_lo] + prefix[..., r_lo, c_lo])

    def total(self, prefix) -> torch.Tensor:
        """(..., R+1, C+1) -> (..., 행동수). 앞쪽 축은 그대로 유지된다."""
        return self._corners(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_hi)

    def legal_mask(self, value_prefix: torch.Tensor) -> torch.Tensor:
        """합이 10 이고 네 변이 모두 비어있지 않은 직사각형 (game.Board 와 동일 판정)."""
        total = self.total(value_prefix)
        top = self._corners(value_prefix, self.r_lo, self.c_lo, self.r_lo + 1, self.c_hi)
        bottom = self._corners(value_prefix, self.r_hi - 1, self.c_lo, self.r_hi, self.c_hi)
        left = self._corners(value_prefix, self.r_lo, self.c_lo, self.r_hi, self.c_lo + 1)
        right = self._corners(value_prefix, self.r_lo, self.c_hi - 1, self.r_hi, self.c_hi)
        return ((total - TARGET_SUM).abs() < 0.5) & (top > 0) & (bottom > 0) & (left > 0) & (right > 0)


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(N_GROUPS, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(N_GROUPS, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class GridEncoder(BaseFeaturesExtractor):
    """관측 -> [셀별 임베딩 | 사과 점유 누적합 | 숫자값 누적합]."""

    def __init__(self, observation_space, width: int = WIDTH,
                 n_blocks: int = N_BLOCKS, embed_dim: int = EMBED_DIM):
        n_channels, rows, cols = observation_space.shape  # type: ignore
        self.cell_dim = embed_dim * rows * cols
        self.prefix_dim = (rows + 1) * (cols + 1)
        super().__init__(observation_space, features_dim=self.cell_dim + 2 * self.prefix_dim)

        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim
        self.register_buffer("digit_values",
                             torch.arange(n_channels, dtype=torch.float32)[None, :, None, None])

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(N_GROUPS, width), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.global_fc = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(2 * width, embed_dim, kernel_size=1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(observations))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        g = self.global_fc(pooled)[:, :, None, None].expand(-1, -1, self.rows, self.cols)
        cells = self.mix(torch.cat([h, g], dim=1))

        occupied = 1.0 - observations[:, 0]
        digits = (observations[:, :10] * self.digit_values[:, :10]).sum(dim=1)

        return torch.cat([
            cells.flatten(1),
            prefix_sum(occupied).flatten(1),
            prefix_sum(digits).flatten(1),
        ], dim=1)


class QuantileRectangleHead(nn.Module):
    """셀 임베딩 -> (배치, 분위수, 행동수)."""

    def __init__(self, rows: int, cols: int, embed_dim: int, index: RectIndex,
                 n_quantiles: int = N_QUANTILES, head_dim: int = HEAD_DIM):
        super().__init__()
        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim
        self.n_quantiles = n_quantiles
        self.index = index

        # 공통 위치 항
        self.proj_tl = nn.Linear(embed_dim, head_dim)
        self.proj_br = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5
        self.region = nn.Conv2d(embed_dim, 1, kernel_size=1)

        self.n_counts = MAX_APPLE_COUNT + 1
        self.register_buffer("shape_idx",
                             (index.r_hi - 1 - index.r_lo) * cols + (index.c_hi - 1 - index.c_lo))
        self.shape_count_bias = nn.Parameter(torch.zeros(rows * cols, self.n_counts))
        self.register_buffer("tl_idx", index.r_lo * cols + index.c_lo)
        self.register_buffer("br_idx", (index.r_hi - 1) * cols + (index.c_hi - 1))

        # 분위수별 항
        self.spread = nn.Conv2d(embed_dim, n_quantiles, kernel_size=1)
        self.level = nn.Parameter(torch.zeros(n_quantiles))

        for layer in (self.proj_tl, self.proj_br):
            nn.init.orthogonal_(layer.weight, gain=0.3)
            nn.init.zeros_(layer.bias)
        for layer in (self.region, self.spread):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, cells: torch.Tensor, occ_prefix: torch.Tensor) -> torch.Tensor:
        flat = cells.flatten(2).transpose(1, 2)                       # (N, R*C, E)
        pair = torch.bmm(self.proj_tl(flat),
                         self.proj_br(flat).transpose(1, 2)) * self.scale
        base = pair[:, self.tl_idx, self.br_idx]                      # (N, A)
        base = base + self.index.total(prefix_sum(self.region(cells).squeeze(1)))

        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        base = base + self.shape_count_bias.view(-1)[self.shape_idx * self.n_counts + count]

        # (N, Q, R+1, C+1) -> (N, Q, A)
        spread = self.index.total(prefix_sum(self.spread(cells)))
        return base[:, None, :] + spread + self.level[None, :, None]


class MaskedQuantileNetwork(QuantileNetwork):
    """sb3_contrib QuantileNetwork 의 MLP 를 직사각형 헤드로 갈아끼우고 마스크를 씌운다."""

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor  # GridEncoder
        self.index = RectIndex(actions)
        self.quantile_net = QuantileRectangleHead(
            fe.rows, fe.cols, fe.embed_dim, self.index, self.n_quantiles, head_dim)  # type: ignore[arg-type]
        self.cell_dim = fe.cell_dim      # type: ignore[assignment]
        self.prefix_dim = fe.prefix_dim  # type: ignore[assignment]
        self.rows, self.cols = fe.rows, fe.cols  # type: ignore[assignment]

    def _split(self, features: torch.Tensor):
        n = features.shape[0]
        cells = features[:, :self.cell_dim].view(n, -1, self.rows, self.cols)
        rest = features[:, self.cell_dim:]
        occ = rest[:, :self.prefix_dim].view(n, self.rows + 1, self.cols + 1)
        value = rest[:, self.prefix_dim:].view(n, self.rows + 1, self.cols + 1)
        return cells, occ, value

    def legal_mask(self, obs) -> torch.Tensor:
        _, _, value_prefix = self._split(self.extract_features(obs, self.features_extractor))
        return self.index.legal_mask(value_prefix)

    def forward(self, obs) -> torch.Tensor:
        cells, occ_prefix, value_prefix = self._split(
            self.extract_features(obs, self.features_extractor))
        quantiles = self.quantile_net(cells, occ_prefix)              # (N, Q, A)
        mask = self.index.legal_mask(value_prefix)[:, None, :]        # (N, 1, A)
        return quantiles.masked_fill(~mask, MASK_FILL)


class MaskedQRDQNPolicy(QRDQNPolicy):
    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        self._actions = actions
        self._head_dim = head_dim
        super().__init__(*args, **kwargs)

    def make_quantile_net(self) -> MaskedQuantileNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQuantileNetwork(**net_args, actions=self._actions,
                                     head_dim=self._head_dim).to(self.device)


POLICY_CLASS = MaskedQRDQNPolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(width=WIDTH, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM),
        net_arch=[],                          # quantile_net 을 통째로 교체하므로 쓰이지 않는다
        n_quantiles=N_QUANTILES,
        normalize_images=False,
        actions=get_all_action(rows, cols),
        head_dim=HEAD_DIM,
    )
