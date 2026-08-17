"""DQN V11a 신경망 — V1.0 구조에 **듀얼링 분해**를 넣었다.

V1.0 과의 차이는 이 한 줄이다.

    (V1.0)   Q(s,a) = head(s,a)
    (V11a)   Q(s,a) = V(s) + [ A(s,a) - mean_{합법 a'} A(s,a') ]

**왜 이 분해인가.** 이 프로젝트가 계속 부딪힌 벽은 문서(§3, V7.0)에 정리된 그대로다.
explained_variance 가 0.983 이라 '이 판이 대충 몇 점짜리인가'는 잘 맞히는데,
정작 필요한 것은 '같은 판에서 갈라진 형제 중 어느 쪽이 나은가'이고 그 차이는
겨우 2점 남짓이다. 숫자로 보면

    점수 std 12.77 -> 보상 단위 0.0788
    EV 0.983      -> 잔차 std = 0.0788 x sqrt(0.017) ~= 0.0103  (약 1.7점)

즉 **공통항이 0.5 규모인데 우리가 배워야 하는 행동항은 0.01 규모**다. 두 개를 한
출력으로 회귀시키면 손실의 99% 가 공통항 몫이고, 행동항은 그 잔차에 묻힌다.
듀얼링은 공통항을 V(s) 라는 별도 경로로 빼내서 헤드가 처음부터 **차이만** 배우게 한다.

QR-DQN(V9a)이 효과 0(-0.09 ± 1.33)이었던 것과 대비된다. 분위수 16개는 **같은
스칼라를 여러 각도로** 볼 뿐이라 이 분해를 해주지 못한다. 이번엔 표현을 나눈다.

**행동항의 평균을 왜 합법수에 대해서만 빼는가.** 7533개 중 실제로 둘 수 있는 것은
보통 3~47개다. 불법 수까지 포함해 평균을 내면 그 평균이 7500여 개의 의미 없는
값에 끌려다녀 V(s) 와 A(s,a) 의 분리가 흐려진다. 마스크는 어차피 순전파마다
계산하므로 비용이 없다.

나머지(GridEncoder, RectangleHead, 신경망 안 마스킹)는 V1.0 과 동일하다.
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
VALUE_HIDDEN = 256   # V(s) 전용 MLP. 듀얼링에서 공통항을 통째로 짊어진다.
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
    """관측 -> [셀별 임베딩 | 사과 점유 누적합 | 숫자값 누적합]. V1.0 과 동일."""

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
        digits = (observations[:, :10] * self.digit_values[:, :10]).sum(dim=1)

        return torch.cat([
            cells.flatten(1),
            prefix_sum(occupied).flatten(1),
            prefix_sum(digits).flatten(1),
        ], dim=1)


class RectangleHead(nn.Module):
    """셀 임베딩 + 누적합에서 7533개 직사각형의 **이점(advantage)** 을 한 번에 만든다.

    구조는 V1.0 의 Q 헤드와 완전히 같다. 바뀐 것은 이 출력이 이제 Q 가 아니라
    A(s,a) 라는 해석뿐이다 (MaskedQNetwork 가 V(s) 를 더하고 평균을 뺀다).
    """

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
        flat = cells.flatten(2).transpose(1, 2)                       # (N, R*C, E)
        pair = torch.bmm(self.proj_tl(flat),
                         self.proj_br(flat).transpose(1, 2)) * self.scale
        adv = pair[:, self.tl_idx, self.br_idx]

        adv = adv + self.index.total(prefix_sum(self.region(cells).squeeze(1)))

        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        return adv + self.shape_count_bias.view(-1)[self.shape_idx * self.n_counts + count]


class StateValueHead(nn.Module):
    """듀얼링의 V(s). 셀 임베딩의 평균/최대 풀링 + 남은 사과 수.

    남은 사과 수를 스칼라로 직접 넣는 이유는 V8c 와 같다 — 그 값이 남은 점수의
    상한이라 가치 예측의 뼈대인데, 풀링된 임베딩에서 복원하게 두는 것은 낭비다.
    마지막 층을 0 으로 초기화해 학습 시작 시 V(s)=0, 즉 V1.0 과 같은 상태에서 출발한다.
    """

    def __init__(self, embed_dim: int, n_cells: int, hidden: int = VALUE_HIDDEN):
        super().__init__()
        self.n_cells = n_cells
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cells: torch.Tensor, occ_prefix: torch.Tensor) -> torch.Tensor:
        flat = cells.flatten(2)                                   # (N, E, R*C)
        remaining = occ_prefix[:, -1, -1:] / self.n_cells         # (N, 1) 남은 사과 수
        pooled = torch.cat([flat.mean(dim=2), flat.amax(dim=2), remaining], dim=1)
        return self.mlp(pooled)                                   # (N, 1)


class MaskedQNetwork(QNetwork):
    """SB3 QNetwork 의 q_net(MLP)을 듀얼링 헤드로 갈아끼우고 마스크를 씌운다."""

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor  # GridEncoder
        self.index = RectIndex(actions)
        self.q_net = RectangleHead(fe.rows, fe.cols, fe.embed_dim, fe.cell_dim,  # type: ignore[arg-type]
                                   self.index, head_dim)
        self.value_head = StateValueHead(fe.embed_dim, fe.rows * fe.cols)  # type: ignore[arg-type]
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

        adv = self.q_net(cells, occ_prefix)                        # (N, A)
        state_value = self.value_head(cells, occ_prefix)           # (N, 1)
        mask = self.index.legal_mask(value_prefix)                 # (N, A)

        # 합법수에 대해서만 평균을 뺀다. 합법수가 0개인 종료 상태는 나눗셈만 막아 두면
        # 되는데, SB3 가 타깃에 (1 - done) 을 곱하므로 그 값 자체는 쓰이지 않는다.
        weight = mask.to(adv.dtype)
        n_legal = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
        adv_mean = (adv * weight).sum(dim=1, keepdim=True) / n_legal

        q_values = state_value + (adv - adv_mean)
        return q_values.masked_fill(~mask, MASK_FILL)


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
