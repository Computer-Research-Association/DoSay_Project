"""DQN V11c 신경망 — **한 수 앞의 결과를 예측하도록 배우는** Q 신경망. (새 시도 ②)

──────────────────────────────────────────────────────────────────────────
출발점: 한 줄짜리 휴리스틱이 12M 스텝 신경망을 이긴다
──────────────────────────────────────────────────────────────────────────
문서에 남아 있는 기준선을 다시 보면 이상한 줄이 하나 있다.

    무작위 합법수                       96.2
    최다 제거 탐욕                      91.0
    작은 넓이 우선 (한 줄 휴리스틱)     108.5
    **1수 앞 탐색 (한 줄 휴리스틱)      118.1**
    DQN V1.0 (12M 스텝, 탐색 없음)      114.5

"수를 둬 보고 남는 합법수를 세는" 한 줄짜리 규칙이 12M 스텝을 먹은 신경망보다
**3.6점 높다.** 신경망이 멍청해서가 아니다. 결정 시점에 정보가 없기 때문이다 —
사각형을 지운 *뒤*의 판을 볼 수 없으니까.

빔서치는 그 정보를 추론 시점에 사서 쓴다. 대가는 판당 0.38초 -> 22.6초(60배)이고
얻는 것은 +7점이다. **V11c 는 그 정보를 학습 시점에 사서 가중치에 넣는다.**
추론은 그대로 1회 forward 다.

──────────────────────────────────────────────────────────────────────────
무엇을 배우게 하는가
──────────────────────────────────────────────────────────────────────────
보조 목표 ① **look(s,a) = 그 수를 둔 뒤 남는 합법수** (행동마다 하나)

    라벨은 손으로 튜닝한 값이 아니라 게임 규칙 그대로다. 사각형을 0 으로 지운
    판의 누적합에 legal_mask 를 씌워 세면 game.Board 와 **완전히 같은 판정**이
    나온다(아래 RectIndex.afterstate_legal_counts). 전부 GPU 텐서 연산이라
    게임 엔진을 부르지 않는다.

보조 목표 ② **survive(s, 칸) = 이 사과가 끝까지 남을 확률** (칸마다 하나, 162개)

    라벨은 에이전트 자신의 에피소드 결말에서 나온다(env.py 가 final_grid 를
    내보낸다). 게임의 회계 구조상 점수 = 162 - 남은 사과 수 이므로, 이것은
    '무엇이 좌초되는가'를 직접 표현하게 만드는 목표다.

**왜 이것이 QR-DQN 과 다른가.** V9a 의 분위수 16개는 짝비교로 -0.09 ± 1.33,
정확히 0 이었다. 분위수는 **같은 스칼라를 여러 각도로 보는 것**이라 상태당
감독 정보가 늘지 않는다. 여기서는 상태당 라벨이

    TD 목표 1개  ->  look 8개(샘플한 합법수) + survive 162개 = 약 170개

로 늘고, 무엇보다 **다른 것을 배운다.**

──────────────────────────────────────────────────────────────────────────
Q 와 어떻게 연결하는가
──────────────────────────────────────────────────────────────────────────
    Q(s,a) = base(s,a) + look_weight · look(s,a).detach()

`look_weight` 는 0 에서 시작하는 학습 파라미터다. **look 을 detach 하는 이유**:
그렇게 해야 look 은 오직 보조 손실(예측 정확도)로만 학습되고, TD 손실은
'그 예측을 얼마나 신뢰할지'(=계수 하나)만 정하게 된다. TD 가 look 을 직접
비틀면 예측기가 아니라 또 하나의 자유로운 Q 항이 되어 버려 실험이 무의미해진다.

이 구조가 env.py 에서 뺀 Φ=0.3·log1p(합법수) 를 대체한다. 다른 점 두 가지:
  - Φ 는 **현재 상태** s 의 함수라 argmax 를 못 바꿨다. look 은 **행동별**이다.
  - 계수 0.3 은 사람이 골랐다. look_weight 는 학습된다.

`survive` 는 Q 에 직접 들어가지 않는다. 공유 트렁크를 통해서만 영향을 준다
(표준적인 보조과제 구성). Q 출력 자체에 손실을 걸면 V10b/V10c 처럼 Q 의 눈금이
망가진다 — train/loss 가 각각 5180배, 953배로 폭주했다.

나머지(GridEncoder, RectangleHead, 신경망 안 마스킹)는 V1.0 과 동일하다.
"""

import math

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
LOOK_HIDDEN = 64
N_GROUPS = 8
MAX_APPLE_COUNT = 10
TARGET_SUM = 10          # 합이 10 이면 제거 가능
MASK_FILL = -1e8         # 불법 수의 Q. -inf 는 NaN 을 부르므로 큰 음수로.

# look 목표를 [0, 1) 로 정규화하는 값. 관측의 밀도 채널(DENSITY_LOG_SCALE)과 같은 눈금.
LOOK_LOG_SCALE = math.log1p(64.0)


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    """(..., R, C) -> (..., R+1, C+1) 0 으로 패딩된 2차원 누적합."""
    padded = F.pad(plane, (1, 0, 1, 0))
    return padded.cumsum(dim=-2).cumsum(dim=-1)


def grid_from_obs(observations: torch.Tensor) -> torch.Tensor:
    """관측의 숫자 원-핫 10채널에서 숫자판을 복원한다. (N, R, C)"""
    digits = torch.arange(10, device=observations.device,
                          dtype=observations.dtype)[None, :, None, None]
    return (observations[:, :10] * digits).sum(dim=1)


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

    # ── 보조 목표 ①의 라벨 생성 ──────────────────────────────────────────
    def afterstate_legal_counts(self, grid: torch.Tensor, action_idx: torch.Tensor,
                                chunk: int = 512) -> torch.Tensor:
        """(B, R, C) 숫자판과 (B, L) 행동 인덱스 -> (B, L) **그 수를 둔 뒤 남는 합법수**.

        게임 엔진을 부르지 않는다. 사각형 안을 0 으로 지운 판의 누적합에 legal_mask
        를 씌워 세는 것이 전부이고, 그 판정은 game.Board._update_valid_actions 와
        정확히 같다 (합 10 + 네 변 비어있지 않음).

        메모리 주의: legal_mask 는 (chunk, 7533) 중간 텐서를 몇 개 만든다. chunk 512
        면 텐서 하나가 약 15MB 다. B x L 이 크면 chunk 를 줄일 것.
        """
        batch, rows, cols = grid.shape
        n_pick = action_idx.shape[1]
        flat = action_idx.reshape(-1)                                    # (M,)
        boards = grid[:, None].expand(batch, n_pick, rows, cols).reshape(-1, rows, cols)

        row_ix = torch.arange(rows, device=grid.device)[None, :, None]
        col_ix = torch.arange(cols, device=grid.device)[None, None, :]
        inside = ((row_ix >= self.r_lo[flat][:, None, None])
                  & (row_ix < self.r_hi[flat][:, None, None])
                  & (col_ix >= self.c_lo[flat][:, None, None])
                  & (col_ix < self.c_hi[flat][:, None, None]))
        after = boards.masked_fill(inside, 0.0)

        counts = []
        for start in range(0, after.shape[0], chunk):
            block = prefix_sum(after[start:start + chunk])
            counts.append(self.legal_mask(block).sum(dim=1))
        return torch.cat(counts).view(batch, n_pick).to(grid.dtype)


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
        digits = grid_from_obs(observations)

        return torch.cat([
            cells.flatten(1),
            prefix_sum(occupied).flatten(1),
            prefix_sum(digits).flatten(1),
        ], dim=1)


class RectangleHead(nn.Module):
    """셀 임베딩 + 누적합에서 7533개 직사각형의 Q 기본항을 한 번에 만든다. V1.0 과 동일."""

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
        q = pair[:, self.tl_idx, self.br_idx]

        q = q + self.index.total(prefix_sum(self.region(cells).squeeze(1)))

        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        return q + self.shape_count_bias.view(-1)[self.shape_idx * self.n_counts + count]


class LookaheadHead(nn.Module):
    """행동마다 '그 수를 둔 뒤 남는 합법수'(정규화된 log1p)를 예측한다.

    세 항의 합으로 만든다. 이렇게 나누지 않으면 학습이 잘 안 된다 —

        look(a) = level(s)              상태 수준   (행동에 무관, 판이 얼마나 비었나)
                + region(a)             위치 의존   (누적합 구간합, 학습됨)
                + bias[shape, count]    기하 x 사과 수

    region 항만 쓰면 곤란한 이유: 그 항은 사각형 **안 칸들의 합**이라 넓이가 클수록
    항의 개수가 늘어난다. 그런데 예측 대상(남는 합법수)은 넓이에 그렇게 비례하지
    않는다. level 이 눈금을, bias 가 기하 보정을 맡아야 region 이 위치 정보만
    담당할 수 있다. (Q 헤드가 shape_count_bias 를 두는 것과 같은 이유다.)

    전부 0 으로 초기화해 학습 시작 시 look = 0 에서 출발한다.
    """

    def __init__(self, rows: int, cols: int, embed_dim: int, index: RectIndex,
                 hidden: int = LOOK_HIDDEN):
        super().__init__()
        self.index = index
        self.n_counts = MAX_APPLE_COUNT + 1
        self.register_buffer("shape_idx",
                             (index.r_hi - 1 - index.r_lo) * cols + (index.c_hi - 1 - index.c_lo))
        self.shape_count_bias = nn.Parameter(torch.zeros(rows * cols, self.n_counts))

        self.level = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )
        self.region = nn.Conv2d(embed_dim, 1, kernel_size=1)

        nn.init.zeros_(self.level[-1].weight)
        nn.init.zeros_(self.level[-1].bias)
        nn.init.zeros_(self.region.weight)
        nn.init.zeros_(self.region.bias)

    def forward(self, cells: torch.Tensor, occ_prefix: torch.Tensor) -> torch.Tensor:
        flat = cells.flatten(2)                                        # (N, E, R*C)
        level = self.level(torch.cat([flat.mean(dim=2), flat.amax(dim=2)], dim=1))   # (N, 1)
        local = self.index.total(prefix_sum(self.region(cells).squeeze(1)))          # (N, A)

        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        bias = self.shape_count_bias.view(-1)[self.shape_idx * self.n_counts + count]
        return level + local + bias


class MaskedQNetwork(QNetwork):
    """SB3 QNetwork 의 q_net(MLP)을 직사각형 헤드로 갈아끼우고, 보조 헤드 둘을 더한다."""

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor  # GridEncoder
        self.index = RectIndex(actions)
        self.q_net = RectangleHead(fe.rows, fe.cols, fe.embed_dim, fe.cell_dim,  # type: ignore[arg-type]
                                   self.index, head_dim)
        self.look_net = LookaheadHead(fe.rows, fe.cols, fe.embed_dim, self.index)  # type: ignore[arg-type]
        self.survive_net = nn.Conv2d(fe.embed_dim, 1, kernel_size=1)  # type: ignore[arg-type]
        # look 을 Q 에 얼마나 반영할지. 0 에서 시작해 TD 손실이 정한다.
        self.look_weight = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.survive_net.weight)
        nn.init.zeros_(self.survive_net.bias)

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

    def heads(self, obs):
        """트렁크를 한 번만 통과시켜 (Q, look, survive 로짓, 합법마스크) 를 모두 낸다.

        학습 루프가 보조 손실을 계산할 때 쓴다. 추론 경로(forward)는 Q 만 쓴다.
        """
        cells, occ_prefix, value_prefix = self._split(
            self.extract_features(obs, self.features_extractor))

        look = self.look_net(cells, occ_prefix)                     # (N, A)
        # look 은 보조 손실로만 학습된다. TD 는 '얼마나 믿을지'(계수)만 정한다.
        q_values = self.q_net(cells, occ_prefix) + self.look_weight * look.detach()

        mask = self.index.legal_mask(value_prefix)
        survive_logit = self.survive_net(cells).squeeze(1)           # (N, R, C)
        return q_values.masked_fill(~mask, MASK_FILL), look, survive_logit, mask

    def forward(self, obs) -> torch.Tensor:
        return self.heads(obs)[0]


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
