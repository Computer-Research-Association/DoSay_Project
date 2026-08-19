"""DQN V14 신경망 — **빔이 찾은 수순을 신경망에 증류한다.** 배포 모드는 셋.

    MODE = "policy"      정책 로짓 argmax.        1회 forward.        0.006초/수
    MODE = "afterstate"  합법수를 지운 판을 평가.  깊이 1 (배치 1회).  0.009초/수
    MODE = "beam"        빔으로 한 판을 끝까지 계획하고 그대로 둔다.    0.05초/수 (W=32)

**가중치는 셋 다 완전히 같다.** 이 상수만 다른 쌍둥이 폴더를 두면 같은 체크포인트가
세 모드로 로드된다 (V9c/V9cB 와 같은 방식이라 재학습이 필요 없다).

──────────────────────────────────────────────────────────────────────────
왜 이 구조인가
──────────────────────────────────────────────────────────────────────────
**알파고도 1회 forward 로는 훨씬 약하다.** AlphaGo Zero 의 정책망 단독은 약
3,000 Elo 인데 MCTS 를 붙이면 약 5,200 이다. "깊은 수 설계" 는 신경망이 아니라
탐색이 만들고, 신경망은 탐색을 좁혀 주는 역할이다.

우리 게임은 바둑보다 불리하기까지 하다. 바둑은 '모양' 이 가치를 결정해 잘
일반화되지만, 이 게임은 숫자 하나만 달라도 최적해가 달라지는 **조합 짜맞추기**다.
실측으로도 형제 수 사이의 실제 가치 차이(~2점)가 잘 학습된 가치망의 오차(~1.7점)와
같은 크기라 원리적으로 형제를 못 가린다.

그래서 V6~V12 처럼 **표현·구조·가치학습을 고쳐 봐야 소용이 없었다**
(V11c 는 1수앞을 r=0.992, 좌초를 89% 로 예측하면서도 109점이었다).

**대신 아직 안 해 본 것이 하나 있다: 대량의 고품질 교사 라벨로 지도학습.**
V13L 이 101점인 것은 교사 라벨이 2.5만 개뿐이었기 때문이고, 게다가 그 교사는
자기 정책으로 롤아웃한 것이라 학생보다 나을 수가 없었다.

**빔은 다르다.** 빔은 *가치* 헤드만 쓰므로 정책과 무관하게 강하다. 실측:

| 평가 | 폭 128 |
|---|---|
| 손으로 쓴 것 (합법수 세기) | 120.3 |
| **학습된 가치 (V12b)** | **126.7** |

**+6.4 가 학습의 몫이다.** 그리고 그 가치는 114점짜리 플레이 데이터로 배운 것이다.
126.7 짜리 데이터로 다시 배우면 어떻게 되는가 — 그것이 이 버전의 질문이다.

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
BEAM_WIDTH = 32
AFTERSTATE_CHUNK = 1024
PLAN_CACHE_LIMIT = 20_000


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
    @torch.no_grad()
    def plan(self, grid: torch.Tensor, width: int) -> tuple[list, int]:
        """빔으로 한 판을 끝까지 두고 **최고 수순**을 [(판, 수), ...] 로 돌려준다.

        판이 정해지면 이 게임에는 무작위성이 없으므로, 처음에 끝까지 계획한 수순을
        그대로 두는 것과 매 수 재계획하는 것이 정확히 같다. 그래서 한 판에 한 번만 돈다.
        """
        index = self.index
        grids = grid[None].clone()
        paths: list[list] = [[]]
        occupied0 = int((grid != 0).sum().item())
        best_path, best_score = [], -1

        for _ in range(self.n_cells):
            masks = index.legal_mask_from_grid(grids)
            alive = masks.any(dim=1)

            for i in (~alive).nonzero(as_tuple=True)[0].tolist():
                score = occupied0 - int((grids[i] != 0).sum().item())
                if score > best_score:
                    best_score, best_path = score, paths[i]
            if not bool(alive.any()):
                break

            keep = alive.nonzero(as_tuple=True)[0]
            grids, masks = grids[keep], masks[keep]
            paths = [paths[i] for i in keep.tolist()]

            state_idx, action_idx = masks.nonzero(as_tuple=True)
            children = index.erase(grids[state_idx], action_idx)

            # 지운 칸 집합이 같으면 점수도 같다. 폭을 중복에 낭비하지 않는다.
            _, uniq = np.unique(children.reshape(children.shape[0], -1).cpu().numpy(),
                                axis=0, return_index=True)
            sel = torch.as_tensor(uniq, device=children.device)
            children, state_idx, action_idx = children[sel], state_idx[sel], action_idx[sel]

            leftover = self.expected_leftover(index.observation(children))
            order = torch.argsort(leftover)[:width]
            parents = state_idx[order].tolist()
            chosen = action_idx[order].tolist()
            paths = [paths[p] + [(grids[p].clone(), a)] for p, a in zip(parents, chosen)]
            grids = children[order].contiguous()

        return best_path, best_score

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
                path, _ = self.plan(grids[i], BEAM_WIDTH)
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
