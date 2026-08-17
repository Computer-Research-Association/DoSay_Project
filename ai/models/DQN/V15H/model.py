"""DQN V15H — **V15 의 쌍둥이.** 이 파일은 손으로 고치지 않는다.

    예산 상한 측정용 (132.91). 폭 2048 + 정책 top-8.
    구조는 V15 와 같고 BEAM_WIDTH=2048, POLICY_TOPK=8 만 다르다.

고칠 일이 있으면 `ai/models/DQN/V15/model.py` 를 고치고 다시 생성한다.
"""

"""DQN V15 신경망 — **정책이 탐색을 좁힌다.** V14 에서 딱 이것이 빠져 있었다.

    MODE = "policy"      정책 로짓 argmax.        1회 forward.
    MODE = "afterstate"  합법수를 지운 판을 평가.  깊이 1.
    MODE = "beam"        빔으로 한 판을 끝까지 계획하고 그대로 둔다.

**가중치는 셋 다 완전히 같다.** 이 상수만 다른 쌍둥이 폴더를 두면 같은 체크포인트가
여러 모드로 로드된다 (V9c/V9cB 와 같은 방식이라 재학습이 필요 없다).

──────────────────────────────────────────────────────────────────────────
V14 에서 무엇이 틀렸는가 (2026-08-12 실측)
──────────────────────────────────────────────────────────────────────────
**(1) 빔이 정책 헤드를 한 번도 쓰지 않았다.** 노드마다 합법수 전부(평균 28.5개)를
가치망에 넣고 상위 W개를 남겼다. 그래서 판당 신경망 호출이

    폭  32 ->  5.0만회 -> 124.80점
    폭 128 -> 20.1만회 -> 127.88점
    폭 256 -> 40.1만회 -> 129.15점

**알파제로가 바둑 한 판에 쓰는 것이 20만회다.** 9x18 사과게임에 그 2배를 쓰고
129점이었다. 알파제로에서 800회 탐색이 강한 이유는 정책망이 볼 곳을 좁혀
주기 때문인데, 그 부분이 아예 없었다. 폭 8배로 +4.35 를 산 것이고, 그것은
방법의 개선이 아니라 컴퓨팅의 소모다.

**(2) 정책 목표가 "빔이 고른 수 하나" 였다.** λ=3 margin 손실은 목표가 모호하면
"모든 로짓을 같게" 만드는 퇴화 해(손실 = λ)가 국소 최적이다. 300k 스텝 내내
2.93 에 앉아 있었고 teacher_agreement 는 0.24 -> 0.11 로 오히려 내려갔다.
알파제로는 이 자리에 **MCTS 방문 분포**를 넣는다 — 하나가 아니라 분포다.

**(3) 가치 손실이 162^2 로 나눠져 있었다.** 정책 손실의 1/4500 이 되어,
워밍스타트한 가치(MAE 0.679)가 600스텝 만에 4.48 로 즉사했다.

──────────────────────────────────────────────────────────────────────────
V15 가 바꾸는 것
──────────────────────────────────────────────────────────────────────────
**정책 목표를 분포로.** 펼친 부모마다 자식들의 가치 순위를 softmax 한 것을
목표로 삼는다 (`POLICY_TEMP`). 이것은 "가치망 + 1수 앞" 이라는 **정책 개선
연산자를 정책으로 증류**하는 것이다 — 알파제로가 MCTS 로 하는 일과 같은 구조다.

**그 정책이 탐색을 좁힌다.** `POLICY_TOPK > 0` 이면 노드마다 정책 상위 k개만
펼친다. 비용이 28W -> (1+k)W 로 준다 (정책 1회 + 가치 k회). k=8 이면 3.1배 싸다.
**정책이 좋아지면 탐색이 비싸지는 게 아니라 싸진다** — 그것이 도약의 조건이다.

학습 중에는 `POLICY_TOPK=0` (전부 펼침) 으로 둔다. 정책이 고른 것만 펼치면
목표가 자기 선택으로 편향된다 (V12 의 자기확인 고리와 같은 함정). 대신
`plan()` 이 `topk_recall` — 정책 상위 k 안에 가치 1등이 들어오는 비율 — 을
매 스텝 기록하므로, **별도 측정 없이 학습 로그만 보고** 배포에서 k 를 얼마까지
줄일 수 있는지 알 수 있다.

**가치 라벨을 죽은 빔 전부에서.** V14 는 최고 수순 하나(판당 55개)만 썼다.
`plan()` 은 이미 모든 빔의 최종 점수를 계산하고 버린다. 그 빔들이 곧 '경로 밖
음성 표본' 이고, 빔이 가치에게 묻는 것이 정확히 그것이다 (V12 이후 계속 남아
있던 자기확인 고리의 정체).

──────────────────────────────────────────────────────────────────────────
빔이 왜 한 판에 한 번이면 되는가
──────────────────────────────────────────────────────────────────────────
**판이 정해지면 이 게임에는 무작위성이 전혀 없다.** 그래서 처음에 끝까지 계획한
수순을 그대로 따라가는 것(open-loop)과 매 수마다 다시 계획하는 것(closed-loop)이
**정확히 같다.** 매 수 재계획하면 25배 비싸기만 하고 얻는 것이 없다.

그래서 `plan()` 이 한 판의 최선 수순을 통째로 만들고, `forward()` 는 그것을
판 -> 수 사전으로 캐시해 두었다가 꺼내 쓴다.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from ai.envs.action_sets import get_all_action
from game.action import Action

# ── V12b 와 반드시 같아야 하는 값 (워밍스타트로 가중치를 물려받는다) ────────
WIDTH = 64
N_BLOCKS = 4
EMBED_DIM = 48
VALUE_HIDDEN = 256
N_GROUPS = 8

HEAD_DIM = 32            # 정책 헤드 (V14 에서 새로 붙는 부분)
MAX_APPLE_COUNT = 10
TARGET_SUM = 10
MASK_FILL = -1e8

N_DIGITS = 10
DENSITY_LOG_SCALE = math.log1p(64.0)   # env.py 와 반드시 같아야 한다
VALID_ACTION_SCALE = 128.0             # env.py 와 반드시 같아야 한다

# ── 배포 모드 (쌍둥이 폴더가 이 두 줄만 바꾼다) ──────────────────────────
MODE = "beam"            # "policy" | "afterstate" | "beam"
BEAM_WIDTH = 2048
AFTERSTATE_CHUNK = 1024
# 빔이 한 스텝에 평가하는 자식 수는 (폭 x 자식수) 다. 폭 2048/top-8 이면 16,384개고,
# 인코더 활성값만 수 GB 라 VRAM 을 넘기면 Windows 가 시스템 RAM 으로 흘려서 갑자기
# 3~4배 느려진다 (2026-08-13 폭 2048 실측: 60판까지 ~75초/판 -> 이후 ~290초/판).
# 잘라서 평가하면 봉우리만 낮아지고 **결과는 완전히 같다** — GroupNorm 도 헤드도
# 표본별로 계산되므로 배치 크기에 의존하지 않는다.
EVAL_CHUNK = 4096
PLAN_CACHE_LIMIT = 20_000

# ── 정책이 탐색을 좁힌다 (V15 의 핵심) ───────────────────────────────────
# V14 의 빔은 노드마다 **합법수 전부**(평균 28.5개)를 가치망에 넣었다. 정책 헤드는
# 탐색에 개입하지 않았다. 그래서 판당 신경망 호출이 W=256 에서 40만 회다 —
# 알파제로가 바둑 한 판에 쓰는 20만 회의 2배를 9x18 사과게임에 쓰고 있었다.
#
# 알파제로에서 800회 탐색이 강한 이유는 **정책망이 볼 곳을 좁혀 주기 때문**이다.
# POLICY_TOPK > 0 이면 노드마다 정책 상위 k개만 펼친다. 비용이 28W -> (1+k)W 로 준다
# (정책 1회 + 가치 k회). k=8 이면 3.1배 싸다.
#
# **학습에서는 0 으로 둔다.** 정책이 고른 것만 펼치면 정책 목표가 자기 선택으로
# 편향된다 (V12 의 자기확인 고리와 같은 함정). 배포용 쌍둥이 폴더만 k를 올린다.
POLICY_TOPK = 8

# 정책 목표 분포의 온도 (사과 단위). 알파제로의 MCTS 방문 분포에 대응하는 것을
# 여기서는 "가치망이 매긴 자식들의 순위" 로 만든다 — softmax(-잔여/T).
# V14 는 최선 하나만 골라 margin 손실을 걸었고, 목표가 모호하면 "모든 로짓을 같게"
# 만드는 퇴화 해(손실 = λ)가 국소 최적이다. 실측 2.93 (λ=3.0) 이 정확히 그것이었다.
POLICY_TEMP = 2.0
LABEL_TOPK = 8           # 정책 목표에 담을 자식 수
LABEL_FILL = 1.0e4       # 자식이 없는 슬롯 (softmax 에서 0 이 된다)
MAX_VALUE_ROWS = 4096    # 판당 가치 라벨 상한. 앞쪽 판은 빔 수만큼 중복된다


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    return F.pad(plane, (1, 0, 1, 0)).cumsum(dim=-2).cumsum(dim=-1)


def grid_from_obs(observations: torch.Tensor) -> torch.Tensor:
    digits = torch.arange(N_DIGITS, device=observations.device,
                          dtype=observations.dtype)[None, :, None, None]
    return (observations[:, :N_DIGITS] * digits).sum(dim=1)


class RectIndex(nn.Module):
    """행동 <-> 직사각형. 합법 판정, 사과 수, 판 지우기, 관측 인코딩.

    전부 `game.Board` / `env._get_obs()` 와 정확히 같아야 한다.
    `ai/verify/verify_v14.py` 가 매 실행마다 대조한다.
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

        width = cols + 1
        self.register_buffer("corner_tl", r1 * width + c1)
        self.register_buffer("corner_tr", r1 * width + (c2 + 1))
        self.register_buffer("corner_bl", (r2 + 1) * width + c1)
        self.register_buffer("corner_br", (r2 + 1) * width + (c2 + 1))
        self.register_buffer("tl_idx", r1 * cols + c1)
        self.register_buffer("br_idx", r2 * cols + c2)
        self.register_buffer("shape_idx", (r2 - r1) * cols + (c2 - c1))

    @staticmethod
    def _area(prefix, r_lo, c_lo, r_hi, c_hi) -> torch.Tensor:
        return (prefix[:, r_hi, c_hi] - prefix[:, r_lo, c_hi]
                - prefix[:, r_hi, c_lo] + prefix[:, r_lo, c_lo])

    def total(self, prefix) -> torch.Tensor:
        return self._area(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_hi)

    def legal_mask_from_grid(self, grids: torch.Tensor) -> torch.Tensor:
        prefix = prefix_sum(grids)
        total = self.total(prefix)
        top = self._area(prefix, self.r_lo, self.c_lo, self.r_lo + 1, self.c_hi)
        bottom = self._area(prefix, self.r_hi - 1, self.c_lo, self.r_hi, self.c_hi)
        left = self._area(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_lo + 1)
        right = self._area(prefix, self.r_lo, self.c_hi - 1, self.r_hi, self.c_hi)
        return ((total - TARGET_SUM).abs() < 0.5) & (top > 0) & (bottom > 0) & (left > 0) & (right > 0)


    def legal_actions_on(self, grid: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """판 하나에서 **지정한 행동들만** 합법인지 본다. (A,) 불리언.

        `legal_mask_from_grid` 는 7,533개 전부를 계산한다. 걷어내기(`_ruin`) 의
        재생 고리는 수순의 수 하나씩만 확인하면 되는데, 그때마다 전부를 계산하면
        판당 57번 x 7,533개가 된다. **4차 배치(2026-08-16)에서 복구 폭을 64분의 1
        로 줄였는데 반복이 1.8배밖에 안 늘어난 원인이 여기였다** — 병목은 복구가
        아니라 이 재생이었다.
        """
        prefix = prefix_sum(grid[None])
        r_lo, c_lo = self.r_lo[actions], self.c_lo[actions]
        r_hi, c_hi = self.r_hi[actions], self.c_hi[actions]
        total = self._area(prefix, r_lo, c_lo, r_hi, c_hi)
        top = self._area(prefix, r_lo, c_lo, r_lo + 1, c_hi)
        bottom = self._area(prefix, r_hi - 1, c_lo, r_hi, c_hi)
        left = self._area(prefix, r_lo, c_lo, r_hi, c_lo + 1)
        right = self._area(prefix, r_lo, c_hi - 1, r_hi, c_hi)
        ok = ((total - TARGET_SUM).abs() < 0.5) & (top > 0) & (bottom > 0) & (left > 0) & (right > 0)
        return ok[0]

    def apple_counts(self, grids: torch.Tensor) -> torch.Tensor:
        return self.total(prefix_sum((grids != 0).to(grids.dtype)))

    def erase(self, grids: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(self.rows, device=grids.device)[None, :, None]
        cols = torch.arange(self.cols, device=grids.device)[None, None, :]
        inside = ((rows >= self.r_lo[action_idx][:, None, None])
                  & (rows < self.r_hi[action_idx][:, None, None])
                  & (cols >= self.c_lo[action_idx][:, None, None])
                  & (cols < self.c_hi[action_idx][:, None, None]))
        return grids.masked_fill(inside, 0.0)

    def observation(self, grids: torch.Tensor) -> torch.Tensor:
        """(M, R, C) -> (M, 13, R, C). env._get_obs() 와 원소 단위로 같아야 한다."""
        m, rows, cols = grids.shape[0], self.rows, self.cols
        weight = self.legal_mask_from_grid(grids).to(grids.dtype)

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
        obs[:, N_DIGITS + 2] = (weight.sum(dim=1) / VALID_ACTION_SCALE).clamp(max=1.0)[:, None, None]
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
    """**V12b 와 완전히 같아야 한다** (워밍스타트로 가중치를 물려받으므로)."""

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
        self.global_fc = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(2 * width, embed_dim, kernel_size=1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(observations))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        g = self.global_fc(pooled)[:, :, None, None].expand(-1, -1, self.rows, self.cols)
        return self.mix(torch.cat([h, g], dim=1)).flatten(1)


class ScalarLeftoverHead(nn.Module):
    """최종 잔여 사과 수(정규화)의 로짓. **V12b 의 헤드와 완전히 같아야 한다.**"""

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
        flat = cells.flatten(2)
        remaining = occupied.flatten(1).sum(dim=1, keepdim=True) / self.n_cells
        pooled = torch.cat([flat.mean(dim=2), flat.amax(dim=2), remaining], dim=1)
        return self.mlp(pooled).squeeze(1)


class RectangleHead(nn.Module):
    """셀 임베딩에서 7533개 로짓. V1.0/V13 의 헤드 그대로다 (여기서는 **정책**)."""

    def __init__(self, rows: int, cols: int, embed_dim: int, index: RectIndex,
                 head_dim: int = HEAD_DIM):
        super().__init__()
        self.index = index
        self.proj_tl = nn.Linear(embed_dim, head_dim)
        self.proj_br = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5
        self.region = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.n_counts = MAX_APPLE_COUNT + 1
        self.shape_count_bias = nn.Parameter(torch.zeros(rows * cols, self.n_counts))

        for layer in (self.proj_tl, self.proj_br):
            nn.init.orthogonal_(layer.weight, gain=0.3)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.region.weight)
        nn.init.zeros_(self.region.bias)

    def forward(self, cells: torch.Tensor, occ_prefix: torch.Tensor) -> torch.Tensor:
        flat = cells.flatten(2).transpose(1, 2)
        pair = torch.bmm(self.proj_tl(flat), self.proj_br(flat).transpose(1, 2)) * self.scale
        logits = pair[:, self.index.tl_idx, self.index.br_idx]
        logits = logits + self.index.total(prefix_sum(self.region(cells).squeeze(1)))
        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        return logits + self.shape_count_bias.view(-1)[
            self.index.shape_idx * self.n_counts + count]


class TeacherDistilledQNetwork(QNetwork):
    """가치 헤드(빔용)와 정책 헤드(배포용)를 함께 가진 신경망."""

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor
        self.rows, self.cols = fe.rows, fe.cols          # type: ignore[assignment]
        self.embed_dim = fe.embed_dim                    # type: ignore[assignment]
        self.n_cells = self.rows * self.cols
        self.index = RectIndex(actions, self.rows, self.cols)
        # 이름을 q_net 으로 두는 이유: V12b 체크포인트의 키와 맞춰 워밍스타트하기 위해서다
        self.q_net = ScalarLeftoverHead(self.embed_dim, self.n_cells)
        self.policy_net = RectangleHead(self.rows, self.cols, self.embed_dim,
                                        self.index, head_dim)
        self._plan_cache: dict[bytes, int] = {}
        # 학습 때만 켠다. 빔이 한 판을 계획할 때 나오는 라벨을 여기에 쌓아 두고
        # train.py 가 꺼내 간다 (배포 때는 항상 비어 있다).
        self.collect_labels = False
        self.pending_labels: list[dict] = []

    # ── 기본 경로 ────────────────────────────────────────────────────────
    def _cells(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return features.view(features.shape[0], self.embed_dim, self.rows, self.cols)

    def policy_logits(self, obs: torch.Tensor) -> torch.Tensor:
        cells = self._cells(obs)
        occupied = (obs[:, 0] < 0.5).to(obs.dtype)
        logits = self.policy_net(cells, prefix_sum(occupied))
        mask = self.index.legal_mask_from_grid(grid_from_obs(obs))
        return logits.masked_fill(~mask, MASK_FILL)

    def expected_leftover(self, obs: torch.Tensor) -> torch.Tensor:
        """이 판에서 끝까지 남을 사과 수의 기대값. 빔이 쓰는 값이다."""
        return torch.sigmoid(self.q_net(self._cells(obs), obs[:, 0] < 0.5)) * self.n_cells

    def legal_mask(self, obs) -> torch.Tensor:
        return self.index.legal_mask_from_grid(grid_from_obs(obs))

    # ── 깊이 1 (애프터스테이트) ──────────────────────────────────────────
    @torch.no_grad()
    def _afterstate_q(self, obs: torch.Tensor) -> torch.Tensor:
        grids = grid_from_obs(obs)
        mask = self.index.legal_mask_from_grid(grids)
        out = torch.full(mask.shape, MASK_FILL, device=obs.device, dtype=obs.dtype)
        state_idx, action_idx = mask.nonzero(as_tuple=True)
        if state_idx.numel() == 0:
            return out
        values = []
        for start in range(0, state_idx.numel(), AFTERSTATE_CHUNK):
            rows = slice(start, start + AFTERSTATE_CHUNK)
            after = self.index.erase(grids[state_idx[rows]], action_idx[rows])
            values.append(self.expected_leftover(self.index.observation(after)))
        out[state_idx, action_idx] = (self.n_cells - torch.cat(values)) / self.n_cells
        return out

    # ── 빔 (교사이자 무거운 배포 모드) ───────────────────────────────────
    def policy_logits_from_grid(self, grids: torch.Tensor) -> torch.Tensor:
        return self.policy_logits(self.index.observation(grids))

    @torch.no_grad()
    def _leftover_chunked(self, children: torch.Tensor) -> torch.Tensor:
        """자식 전부의 예상 잔여. 큰 폭에서 VRAM 봉우리를 자른다 (결과 동일)."""
        if children.shape[0] <= EVAL_CHUNK:
            return self.expected_leftover(self.index.observation(children))
        parts = [self.expected_leftover(self.index.observation(children[s:s + EVAL_CHUNK]))
                 for s in range(0, children.shape[0], EVAL_CHUNK)]
        return torch.cat(parts)

    @torch.no_grad()
    def plan(self, grid: torch.Tensor, width: int, topk: int | None = None,
             collect: bool = False, evaluate=None) -> tuple[list, int, dict | None]:
        """빔으로 한 판을 끝까지 두고 (최고 수순, 점수, 학습 라벨) 을 돌려준다.

        판이 정해지면 무작위성이 없으므로 open-loop 계획과 매 수 재계획이 같다.
        그래서 한 판에 한 번만 돈다.

        `topk > 0` 이면 노드마다 **정책 상위 k개만** 펼친다 (배포용, 비용 k/28).
        `collect` 면 학습 라벨을 함께 모은다.
        `evaluate` 로 가치 대신 다른 평가기를 넣을 수 있다 (자식 판 -> 예상 잔여).
        **AI 트랙의 기여분을 재는 대조군이 이걸 쓴다** — 같은 탐색을 손 평가로
        돌린 것과의 차이가 곧 학습의 몫이다 (빔에서 +6.48).

            가치 — **죽은 빔 전부**의 경로 위 모든 판에 "여기서 앞으로 얻은 점수".
                   V14 는 최고 수순 하나(판당 55개)만 썼다. 버려진 빔이 곧
                   '경로 밖 음성 표본' 이고, 빔이 가치에게 묻는 것이 정확히 그것이다.
            정책 — 펼친 부모마다 자식들의 가치 순위를 softmax 한 분포. 알파제로의
                   MCTS 방문 분포에 대응한다. "가치망 + 1수 앞" 이라는 개선
                   연산자를 정책으로 증류하는 것이고, 이것이 좋아지면 top-k
                   가지치기가 안전해진다 = **탐색이 싸진다.**
        """
        index = self.index
        topk = POLICY_TOPK if topk is None else topk
        n_actions = int(index.r_lo.numel())
        k_label = min(LABEL_TOPK, n_actions)
        evaluate = evaluate or self._leftover_chunked
        grids = grid[None].clone()
        occupied0 = int((grid != 0).sum().item())
        pol_grids, pol_actions, pol_probs = [], [], []
        recall_hit = recall_total = 0

        # ── 빔을 **부모 포인터 트리**로 들고 다닌다 ──────────────────────
        # 예전에는 빔마다 [(판, 수), ...] 리스트를 통째로 복사했다. 폭 1024 x 깊이 55
        # 면 판당 5.6만 개의 작은 GPU 텐서 클론 + 155만 번의 리스트 원소 복사가 되고,
        # 100판이면 560만 개다. 캐싱 할당자가 단편화되면서 **판이 갈수록 느려졌다**
        # (실측: 38.7초/판 -> 218.5초/판, 5.6배).
        #
        # 깊이마다 (판 묶음, 부모 인덱스, 수) 텐서 세 개만 들면 스텝당 O(W) 이고
        # 클론이 0개다. 최고 수순은 끝에서 한 번 거슬러 올라가 만든다.
        lv_grids = [grids]                                  # 깊이 d 의 판 묶음
        lv_parent: list[torch.Tensor | None] = [None]        # -> 깊이 d-1 의 인덱스
        lv_action: list[torch.Tensor | None] = [None]
        # "이 노드에서 빔이 실제로 낸 최고 점수". 잎에서 채우고 위로 max 로 올린다.
        lv_best = [grids.new_full((1,), -float("inf"))]
        best_score, best_leaf = -1, (0, 0)

        for depth in range(self.n_cells):
            masks = index.legal_mask_from_grid(grids)
            alive = masks.any(dim=1)

            dead = (~alive).nonzero(as_tuple=True)[0]
            if dead.numel():
                scores = occupied0 - (grids[dead] != 0).flatten(1).sum(dim=1)
                lv_best[depth][dead] = scores.to(lv_best[depth].dtype)
                top = int(scores.argmax().item())
                if int(scores[top].item()) > best_score:
                    best_score = int(scores[top].item())
                    best_leaf = (depth, int(dead[top].item()))
            if not bool(alive.any()):
                break

            kept = alive.nonzero(as_tuple=True)[0]
            g, m = grids[kept], masks[kept]

            logits = (self.policy_logits_from_grid(g)
                      if (topk > 0 or collect) else None)
            if topk > 0:
                # 로짓은 불법수가 -1e8 이라 topk 가 합법수를 먼저 집는다.
                # 합법수가 k개보다 적은 노드는 & 로 걸러진다.
                narrowed = torch.zeros_like(m)
                narrowed.scatter_(1, logits.topk(min(topk, n_actions), dim=1).indices, True)
                m = m & narrowed

            state_idx, action_idx = m.nonzero(as_tuple=True)
            children = index.erase(g[state_idx], action_idx)

            # 지운 칸 집합이 같으면 점수도 같다. 폭을 중복에 낭비하지 않는다.
            _, uniq = np.unique(children.reshape(children.shape[0], -1).cpu().numpy(),
                                axis=0, return_index=True)
            sel = torch.as_tensor(uniq, device=children.device)
            children, state_idx, action_idx = children[sel], state_idx[sel], action_idx[sel]

            leftover = evaluate(children)

            if collect:
                # 부모마다 자식들의 가치를 dense 로 흩뿌린 뒤 상위 k개를 목표 분포로.
                dense = g.new_full((g.shape[0], n_actions), LABEL_FILL)
                dense[state_idx, action_idx] = leftover
                # 중복 제거로 자식을 전부 잃은 부모는 라벨이 될 수 없다.
                has = (dense < LABEL_FILL).any(dim=1)
                if bool(has.any()):
                    top = (-dense[has]).topk(k_label, dim=1)
                    pol_grids.append(g[has])
                    pol_actions.append(top.indices)
                    pol_probs.append(torch.softmax(top.values / POLICY_TEMP, dim=1))
                    # 정책 상위 k 안에 **가치 1등**이 들어오는가 = top-k 의 안전성.
                    # 이 값이 1 에 가까워지면 배포에서 폭 대신 k 를 줄일 수 있다.
                    ranked = logits[has].topk(k_label, dim=1).indices
                    hit = (ranked == top.indices[:, :1]).any(dim=1)
                    recall_hit += int(hit.sum().item())
                    recall_total += int(hit.numel())

            order = torch.argsort(leftover)[:width]
            grids = children[order].contiguous()
            lv_grids.append(grids)
            lv_parent.append(kept[state_idx[order]])       # 깊이 d 의 **원본** 인덱스
            lv_action.append(action_idx[order])
            lv_best.append(grids.new_full((grids.shape[0],), -float("inf")))

        best_path = self._walk_back(lv_grids, lv_parent, lv_action, best_leaf)
        labels = None
        if collect:
            labels = self._collect_labels(lv_grids, lv_parent, lv_best, occupied0,
                                          pol_grids, pol_actions, pol_probs,
                                          recall_hit, recall_total)
        return best_path, best_score, labels

    @staticmethod
    def _walk_back(lv_grids, lv_parent, lv_action, leaf) -> list:
        """(깊이, 인덱스) 에서 뿌리까지 거슬러 올라가 [(판, 수), ...] 를 만든다."""
        depth, node = leaf
        parents = [None if p is None else p.cpu().numpy() for p in lv_parent]
        actions = [None if a is None else a.cpu().numpy() for a in lv_action]
        path = []
        while depth > 0:
            p, a = int(parents[depth][node]), int(actions[depth][node])
            path.append((lv_grids[depth - 1][p], a))
            depth, node = depth - 1, p
        path.reverse()
        return path

    @staticmethod
    def _collect_labels(lv_grids, lv_parent, lv_best, occupied0, pol_grids,
                        pol_actions, pol_probs, recall_hit, recall_total) -> dict | None:
        """라벨을 CPU 텐서로 묶는다. 판(grid)만 들고 obs 는 학습 때 GPU 에서 만든다
        (obs 는 13채널이라 판의 13배다).

        가치 목표는 **잎 점수를 트리 위로 max 로 올린 것** = "빔이 이 판에서 실제로
        낸 최고 점수". 예전처럼 잎마다 경로를 통째로 쏟으면 앞쪽 판이 빔 수만큼
        중복돼 초반 상태에 가중치가 쏠렸는데, max 백업은 노드마다 정확히 한 줄이다.
        """
        if not pol_grids:
            return None
        for depth in range(len(lv_best) - 1, 0, -1):
            lv_best[depth - 1].scatter_reduce_(0, lv_parent[depth], lv_best[depth],
                                               reduce="amax", include_self=True)

        boards, remains = [], []
        for depth, (g, best) in enumerate(zip(lv_grids, lv_best)):
            ok = torch.isfinite(best)                 # 잎으로 이어지지 못한 노드는 뺀다
            if not bool(ok.any()):
                continue
            kept_g = g[ok]
            got = occupied0 - (kept_g != 0).flatten(1).sum(dim=1)
            boards.append(kept_g)
            remains.append(best[ok] - got)
        if not boards:
            return None

        stacked, remain = torch.cat(boards), torch.cat(remains)
        if stacked.shape[0] > MAX_VALUE_ROWS:
            pick = torch.randperm(stacked.shape[0], device=stacked.device)[:MAX_VALUE_ROWS]
            stacked, remain = stacked[pick], remain[pick]

        return {
            "value_grids": stacked.to(torch.int8).cpu(),
            "value_remain": remain.to(torch.float32).cpu(),
            "policy_grids": torch.cat(pol_grids).to(torch.int8).cpu(),
            "policy_actions": torch.cat(pol_actions).to(torch.int16).cpu(),
            "policy_probs": torch.cat(pol_probs).cpu(),
            "topk_recall": (recall_hit / recall_total) if recall_total else float("nan"),
        }

    @torch.no_grad()
    def _beam_q(self, obs: torch.Tensor) -> torch.Tensor:
        grids = grid_from_obs(obs)
        out = torch.full((obs.shape[0], self.index.r_lo.numel()), MASK_FILL,
                         device=obs.device, dtype=obs.dtype)
        for i in range(obs.shape[0]):
            key = grids[i].to(torch.int8).cpu().numpy().tobytes()
            if key not in self._plan_cache:
                if len(self._plan_cache) > PLAN_CACHE_LIMIT:
                    self._plan_cache.clear()
                path, _, labels = self.plan(grids[i], BEAM_WIDTH,
                                            collect=self.collect_labels)
                if labels is not None:
                    self.pending_labels.append(labels)
                for board, action in path:
                    self._plan_cache[board.to(torch.int8).cpu().numpy().tobytes()] = action
            action = self._plan_cache.get(key)
            if action is None:      # 계획이 없는 판(빔이 못 도달) -> 정책으로 폴백
                return self.policy_logits(obs)
            out[i, action] = 1.0
        return out

    def forward(self, obs) -> torch.Tensor:
        if MODE == "policy":
            return self.policy_logits(obs)
        if MODE == "afterstate":
            return self._afterstate_q(obs)
        return self._beam_q(obs)


class TeacherDistilledPolicy(DQNPolicy):
    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        self._actions = actions
        self._head_dim = head_dim
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> TeacherDistilledQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return TeacherDistilledQNetwork(**net_args, actions=self._actions,
                                        head_dim=self._head_dim).to(self.device)


POLICY_CLASS = TeacherDistilledPolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(width=WIDTH, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM),
        net_arch=[],
        normalize_images=False,
        actions=get_all_action(rows, cols),
        head_dim=HEAD_DIM,
    )
