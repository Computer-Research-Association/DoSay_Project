"""DQN V12b 신경망 — **V12a 의 대조군. 가치 헤드만 스칼라로 바꿨다.**

V12a 와 다른 것은 딱 하나다.

    (V12a)  칸마다 '이 사과가 끝까지 남을 확률' 162개를 예측하고, 그 합을 가치로 쓴다.
    (V12b)  '최종 잔여 사과 수' **스칼라 하나**를 바로 회귀한다.

애프터스테이트 평가도, 몬테카를로 라벨도, 트렁크도, 하이퍼파라미터도 전부 같다.

**무엇을 가리는 실험인가.** probe_v11c 에서 잔존맵을 애프터스테이트에 씌우면
Q argmax 대비 +4.50 이 나왔는데, 그 이득이 어디서 왔는지가 두 갈래다.

    (가) **조밀한 감독** — 상태당 라벨이 1개가 아니라 162개라서
    (나) **애프터스테이트 평가** — 예측 대신 실제 판을 만들어 재서

V12a 만 돌리면 둘을 구분할 수 없다. V12b 는 (가)만 빼고 (나)는 그대로 둔다.
그래서 **V12a − V12b 가 곧 '조밀한 감독'의 순효과**이고,
**V12b − V1.0(114.49) 이 '애프터스테이트 평가'의 순효과**다.

이 프로젝트가 V4 부터 계속 아쉬웠던 것이 이런 대조군이다. V9c 는 "이긴 조합 둘을
합쳤는데 최저" 였고, V11a 는 변인이 둘이라 귀속이 안 됐다. 이번엔 한 변인만 다르다.

아래는 V12a model.py 의 설명 그대로다 (구조가 같으므로).

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
VALUE_HIDDEN = 256       # 스칼라 가치망. V8c 의 가치망(384x2)과 같은 계열
N_GROUPS = 8
MASK_FILL = -1e8

N_DIGITS = 10
DENSITY_LOG_SCALE = math.log1p(64.0)   # env.py 와 반드시 같아야 한다
VALID_ACTION_SCALE = 128.0             # env.py 와 반드시 같아야 한다
TARGET_SUM = 10

AFTERSTATE_CHUNK = 1024  # 애프터스테이트를 한 번에 몇 개씩 인코딩할지 (메모리 조절)


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    """(..., R, C) -> (..., R+1, C+1) 0 패딩 2차원 누적합."""
    padded = F.pad(plane, (1, 0, 1, 0))
    return padded.cumsum(dim=-2).cumsum(dim=-1)


def grid_from_obs(observations: torch.Tensor) -> torch.Tensor:
    """관측의 숫자 원-핫 10채널에서 숫자판을 복원한다. (N, R, C)"""
    digits = torch.arange(N_DIGITS, device=observations.device,
                          dtype=observations.dtype)[None, :, None, None]
    return (observations[:, :N_DIGITS] * digits).sum(dim=1)


class RectIndex(nn.Module):
    """행동 <-> 직사각형. 합법 판정, 애프터스테이트 생성, 관측 인코딩을 모두 담당한다.

    여기 있는 계산은 전부 `game.Board` / `env._get_obs()` 와 **정확히 같아야** 한다.
    한 곳이라도 어긋나면 신경망이 학습 때 본 적 없는 입력을 받게 되는데, 그건
    조용히 점수만 깎는 종류의 버그다. verify_v12.py 가 매번 대조한다.
    """

    def __init__(self, actions: list[Action], rows: int, cols: int):
        super().__init__()
        self.rows, self.cols = rows, cols
        r1 = torch.tensor([a.top_left[0] for a in actions], dtype=torch.long)
        c1 = torch.tensor([a.top_left[1] for a in actions], dtype=torch.long)
        r2 = torch.tensor([a.bottom_right[0] for a in actions], dtype=torch.long)
        c2 = torch.tensor([a.bottom_right[1] for a in actions], dtype=torch.long)
        for name, value in (("r_lo", r1), ("c_lo", c1), ("r_hi", r2 + 1), ("c_hi", c2 + 1)):
            self.register_buffer(name, value)

        # 옵션 밀도용 2차원 차분 좌표 (env._option_density 와 같은 네 모서리)
        width = cols + 1
        self.register_buffer("corner_tl", r1 * width + c1)
        self.register_buffer("corner_tr", r1 * width + (c2 + 1))
        self.register_buffer("corner_bl", (r2 + 1) * width + c1)
        self.register_buffer("corner_br", (r2 + 1) * width + (c2 + 1))

    # ── 합법 판정 ────────────────────────────────────────────────────────
    @staticmethod
    def _area(prefix, r_lo, c_lo, r_hi, c_hi) -> torch.Tensor:
        return (prefix[:, r_hi, c_hi] - prefix[:, r_lo, c_hi]
                - prefix[:, r_hi, c_lo] + prefix[:, r_lo, c_lo])

    def total(self, prefix) -> torch.Tensor:
        return self._area(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_hi)

    def legal_mask_from_grid(self, grids: torch.Tensor) -> torch.Tensor:
        """(M, R, C) 숫자판 -> (M, 7533) bool. game.Board 와 같은 판정."""
        prefix = prefix_sum(grids)
        total = self.total(prefix)
        top = self._area(prefix, self.r_lo, self.c_lo, self.r_lo + 1, self.c_hi)
        bottom = self._area(prefix, self.r_hi - 1, self.c_lo, self.r_hi, self.c_hi)
        left = self._area(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_lo + 1)
        right = self._area(prefix, self.r_lo, self.c_hi - 1, self.r_hi, self.c_hi)
        return ((total - TARGET_SUM).abs() < 0.5) & (top > 0) & (bottom > 0) & (left > 0) & (right > 0)

    # ── 애프터스테이트 ───────────────────────────────────────────────────
    def erase(self, grids: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """(M, R, C) 판마다 action_idx[i] 사각형을 0 으로 지운다."""
        rows = torch.arange(self.rows, device=grids.device)[None, :, None]
        cols = torch.arange(self.cols, device=grids.device)[None, None, :]
        inside = ((rows >= self.r_lo[action_idx][:, None, None])
                  & (rows < self.r_hi[action_idx][:, None, None])
                  & (cols >= self.c_lo[action_idx][:, None, None])
                  & (cols < self.c_hi[action_idx][:, None, None]))
        return grids.masked_fill(inside, 0.0)

    # ── 관측 인코딩 (env._get_obs 와 동일해야 한다) ───────────────────────
    def observation(self, grids: torch.Tensor) -> torch.Tensor:
        """(M, R, C) 숫자판 -> (M, 13, R, C) 관측."""
        m = grids.shape[0]
        rows, cols = self.rows, self.cols
        mask = self.legal_mask_from_grid(grids)
        weight = mask.to(grids.dtype)

        # 각 칸을 덮는 합법수 개수 — env 와 같은 2차원 차분 + 누적합
        diff = grids.new_zeros(m, (rows + 1) * (cols + 1))
        diff.index_add_(1, self.corner_tl, weight)
        diff.index_add_(1, self.corner_tr, -weight)
        diff.index_add_(1, self.corner_bl, -weight)
        diff.index_add_(1, self.corner_br, weight)
        density = diff.view(m, rows + 1, cols + 1).cumsum(1).cumsum(2)[:, :rows, :cols]

        digits = torch.arange(N_DIGITS, device=grids.device, dtype=grids.dtype)
        obs = grids.new_zeros(m, N_DIGITS + 3, rows, cols)
        obs[:, :N_DIGITS] = (grids.round()[:, None] == digits[None, :, None, None]).to(grids.dtype)

        peak = density.amax(dim=(1, 2), keepdim=True)
        obs[:, N_DIGITS] = torch.where(peak > 0, density / peak.clamp(min=1.0),
                                       torch.zeros_like(density))
        obs[:, N_DIGITS + 1] = (torch.log1p(density) / DENSITY_LOG_SCALE).clamp(max=1.0)
        n_legal = weight.sum(dim=1) / VALID_ACTION_SCALE
        obs[:, N_DIGITS + 2] = n_legal.clamp(max=1.0)[:, None, None]
        return obs


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
    """관측 -> 셀별 임베딩. V1.0 과 같은 트렁크지만 누적합은 싣지 않는다.

    V1.0 은 행동 헤드가 직사각형 구간합을 써야 해서 누적합을 특징에 실어 보냈다.
    V12 에는 행동 헤드가 없다 (칸별 예측만 한다). 그래서 셀 임베딩만 내보낸다.
    """

    def __init__(self, observation_space, width: int = WIDTH,
                 n_blocks: int = N_BLOCKS, embed_dim: int = EMBED_DIM):
        n_channels, rows, cols = observation_space.shape  # type: ignore
        self.cell_dim = embed_dim * rows * cols
        super().__init__(observation_space, features_dim=self.cell_dim)

        self.rows, self.cols, self.embed_dim = rows, cols, embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(N_GROUPS, width), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        # 전역 브랜치: 9x18 판은 잔차 블록의 수용영역보다 넓다
        self.global_fc = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(2 * width, embed_dim, kernel_size=1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(observations))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        g = self.global_fc(pooled)[:, :, None, None].expand(-1, -1, self.rows, self.cols)
        return self.mix(torch.cat([h, g], dim=1)).flatten(1)


class ScalarLeftoverHead(nn.Module):
    """판 하나에 대해 '최종 잔여 사과 수' 스칼라 하나를 낸다.

    V12a 의 잔존맵과 **표현력이 아니라 감독의 밀도**만 다르게 하려고 만든 헤드다.
    셀 임베딩을 평균/최대 풀링해서 받고, 남은 사과 수를 스칼라로 직접 넣어 준다
    (남은 사과 수는 잔여의 상한이라 예측의 뼈대인데 풀링에서 복원하게 두면 낭비다 —
    V8c 의 가치망과 같은 근거).

    출력은 [0, 1] 로 정규화된 잔여 비율이다. 마지막 층을 0 으로 초기화해
    학습 시작 시 0.5 (=칸의 절반이 남는다)에서 출발한다 — V12a 의 잔존맵이
    sigmoid(0)=0.5 에서 출발하는 것과 같은 지점이다.
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

    def forward(self, cells: torch.Tensor, occupied: torch.Tensor) -> torch.Tensor:
        flat = cells.flatten(2)                                     # (M, E, R*C)
        remaining = occupied.flatten(1).sum(dim=1, keepdim=True) / self.n_cells
        pooled = torch.cat([flat.mean(dim=2), flat.amax(dim=2), remaining], dim=1)
        return self.mlp(pooled).squeeze(1)                          # (M,) 로짓


class AfterstateQNetwork(QNetwork):
    """Q 를 예측하지 않는다. 후보 수마다 판을 지워 보고 그 판의 가치를 잰다.

    V12a 와 같은 구조이고 가치 헤드만 스칼라다.
    """

    def __init__(self, *args, actions: list[Action] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor  # GridEncoder
        self.rows, self.cols = fe.rows, fe.cols          # type: ignore[assignment]
        self.embed_dim = fe.embed_dim                    # type: ignore[assignment]
        self.n_cells = self.rows * self.cols
        self.index = RectIndex(actions, self.rows, self.cols)
        self.q_net = ScalarLeftoverHead(self.embed_dim, self.n_cells)   # 이름은 SB3 관례

    # ── 스칼라 가치 ──────────────────────────────────────────────────────
    def leftover_logit(self, obs: torch.Tensor) -> torch.Tensor:
        """정규화된 최종 잔여 비율의 로짓. (M,) 학습이 쓰는 경로."""
        features = self.extract_features(obs, self.features_extractor)
        cells = features.view(features.shape[0], self.embed_dim, self.rows, self.cols)
        return self.q_net(cells, obs[:, 0] < 0.5)

    def expected_leftover(self, obs: torch.Tensor) -> torch.Tensor:
        """이 판에서 끝까지 남을 사과 수의 기대값. (M,)"""
        return torch.sigmoid(self.leftover_logit(obs)) * self.n_cells

    def legal_mask(self, obs) -> torch.Tensor:
        return self.index.legal_mask_from_grid(grid_from_obs(obs))

    # ── 행동 선택 ────────────────────────────────────────────────────────
    def forward(self, obs) -> torch.Tensor:
        """(N, 7533). 합법수는 '그 수를 두면 최종 몇 점이 될까 / 162', 나머지는 -1e8.

        점수 = 162 − 잔여 이므로 잔여를 최소화하는 것이 곧 점수를 최대화하는 것이다.
        먹은 사과 수는 애프터스테이트에 이미 반영돼 있어 따로 더할 필요가 없다.
        """
        grids = grid_from_obs(obs)
        mask = self.index.legal_mask_from_grid(grids)
        q_values = torch.full(mask.shape, MASK_FILL, device=obs.device, dtype=obs.dtype)

        state_idx, action_idx = mask.nonzero(as_tuple=True)
        if state_idx.numel() == 0:
            return q_values                                   # 종료 상태

        leftovers = []
        for start in range(0, state_idx.numel(), AFTERSTATE_CHUNK):
            rows = slice(start, start + AFTERSTATE_CHUNK)
            after = self.index.erase(grids[state_idx[rows]], action_idx[rows])
            leftovers.append(self.expected_leftover(self.index.observation(after)))
        leftover = torch.cat(leftovers)

        q_values[state_idx, action_idx] = (self.n_cells - leftover) / self.n_cells
        return q_values


class AfterstatePolicy(DQNPolicy):
    def __init__(self, *args, actions: list[Action] | None = None, **kwargs):
        self._actions = actions
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> AfterstateQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return AfterstateQNetwork(**net_args, actions=self._actions).to(self.device)


POLICY_CLASS = AfterstatePolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(width=WIDTH, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM),
        net_arch=[],                          # q_net 을 통째로 교체하므로 쓰이지 않는다
        normalize_images=False,
        actions=get_all_action(rows, cols),
    )
