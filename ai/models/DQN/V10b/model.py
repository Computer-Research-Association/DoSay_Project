"""DQN V1.0 신경망 — 마스킹을 신경망 안에서 직접 만든다.

sb3 의 DQN 은 액션 마스킹을 지원하지 않는다. 그런데 이 게임의 합법 판정은
**관측만으로 정확히 계산된다**: 직사각형 안 숫자의 합이 10 이고, 네 변이 모두
비어있지 않으면 합법이다. 둘 다 누적합의 모서리 뺄셈이라 7533개 행동 전부를
gather 몇 번으로 판정할 수 있다.

그래서 관측에 마스크를 실어 보내는(7533칸을 더 붙이는) 대신, Q 신경망이
매 순전파마다 마스크를 스스로 만들어 불법 수의 Q 를 -1e8 로 눌러 버린다.
이 방식의 이점:

  - argmax 가 자동으로 합법수만 고른다 (_predict, 그리고 타깃의 max 둘 다)
  - 관측 규격이 MaskablePPO 쪽 버전들과 똑같이 유지된다
  - 종료 상태는 전부 -1e8 이 되지만 SB3 가 (1 - done) 을 곱하므로 영향이 없다

마스크 계산은 game/board.py 의 결과와 무작위 판 30개에서 완전히 일치함을 확인했다.

Q 헤드는 MaskablePPO 쪽과 같은 RectangleHead 다. 꼭짓점 상호작용 + 내부 요약 +
(기하 x 사과 수) 사전지식의 합으로 7533개 Q 를 한 번에 만든다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from ai.envs.action_sets import get_all_action
from game.action import Action

WIDTH = 64
N_BLOCKS = 4
EMBED_DIM = 48
HEAD_DIM = 32
N_GROUPS = 8
MAX_APPLE_COUNT = 10
TARGET_SUM = 10          # 합이 10 이면 제거 가능
MASK_FILL = -1e8         # 불법 수의 Q. -inf 는 NaN 을 부르므로 큰 음수로.


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    """(N, R, C) -> (N, R+1, C+1) 0 으로 패딩된 2차원 누적합."""
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
    def area(prefix, r_lo, c_lo, r_hi, c_hi) -> torch.Tensor:
        return (prefix[:, r_hi, c_hi] - prefix[:, r_lo, c_hi]
                - prefix[:, r_hi, c_lo] + prefix[:, r_lo, c_lo])

    def total(self, prefix) -> torch.Tensor:
        return self.area(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_hi)

    def legal_mask(self, value_prefix: torch.Tensor) -> torch.Tensor:
        """합이 10 이고 네 변이 모두 비어있지 않은 직사각형 (game.Board 와 동일 판정)."""
        total = self.total(value_prefix)
        top = self.area(value_prefix, self.r_lo, self.c_lo, self.r_lo + 1, self.c_hi)
        bottom = self.area(value_prefix, self.r_hi - 1, self.c_lo, self.r_hi, self.c_hi)
        left = self.area(value_prefix, self.r_lo, self.c_lo, self.r_hi, self.c_lo + 1)
        right = self.area(value_prefix, self.r_lo, self.c_hi - 1, self.r_hi, self.c_hi)
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

        occupied = 1.0 - observations[:, 0]                  # 0번 평면 = 빈칸
        # 0~9 원-핫에 0..9 를 곱해 더하면 숫자값 평면이 된다 (10번 이후 채널은 밀도라 제외)
        digits = (observations[:, :10] * self.digit_values[:, :10]).sum(dim=1)

        return torch.cat([
            cells.flatten(1),
            prefix_sum(occupied).flatten(1),
            prefix_sum(digits).flatten(1),
        ], dim=1)


class RectangleHead(nn.Module):
    """셀 임베딩 + 누적합에서 7533개 직사각형의 Q 를 한 번에 만든다."""

    def __init__(self, rows: int, cols: int, embed_dim: int, cell_dim: int,
                 index: RectIndex, head_dim: int = HEAD_DIM):
        super().__init__()
        self.rows, self.cols, self.embed_dim, self.cell_dim = rows, cols, embed_dim, cell_dim
        self.index = index

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

        for layer in (self.proj_tl, self.proj_br):
            nn.init.orthogonal_(layer.weight, gain=0.3)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.region.weight)
        nn.init.zeros_(self.region.bias)

    def forward(self, cells: torch.Tensor, occ_prefix: torch.Tensor) -> torch.Tensor:
        n = cells.shape[0]
        flat = cells.flatten(2).transpose(1, 2)                       # (N, R*C, E)
        pair = torch.bmm(self.proj_tl(flat),
                         self.proj_br(flat).transpose(1, 2)) * self.scale
        q = pair[:, self.tl_idx, self.br_idx]

        q = q + self.index.total(prefix_sum(self.region(cells).squeeze(1)))

        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        return q + self.shape_count_bias.view(-1)[self.shape_idx * self.n_counts + count]


class MaskedQNetwork(QNetwork):
    """SB3 QNetwork 의 q_net(MLP)을 RectangleHead 로 갈아끼우고 마스크를 씌운다."""

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor  # GridEncoder
        self.index = RectIndex(actions)
        self.q_net = RectangleHead(fe.rows, fe.cols, fe.embed_dim, fe.cell_dim,  # type: ignore[arg-type]
                                   self.index, head_dim)
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
        q_values = self.q_net(cells, occ_prefix)
        return q_values.masked_fill(~self.index.legal_mask(value_prefix), MASK_FILL)


class MaskedDQNPolicy(DQNPolicy):
    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        self._actions = actions
        self._head_dim = head_dim
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> MaskedQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args, actions=self._actions,
                              head_dim=self._head_dim).to(self.device)


POLICY_CLASS = MaskedDQNPolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(width=WIDTH, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM),
        net_arch=[],                          # q_net 을 통째로 교체하므로 쓰이지 않는다
        normalize_images=False,
        actions=get_all_action(rows, cols),
        head_dim=HEAD_DIM,
    )
