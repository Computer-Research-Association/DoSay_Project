"""DQN V13 신경망 — **평가를 학습하지 않는다. 자기 정책으로 끝까지 두어 본다.**

    행동 선택:  후보 수마다 정책으로 **끝까지 롤아웃**해서 최종 점수가 가장 높은 수
    학습:       그 선택을 정답으로 정책 헤드에 margin 손실 (정책반복)

──────────────────────────────────────────────────────────────────────────
왜 이 설계인가 — V6~V12 가 끝낸 줄기와, 그 대신 열린 줄기
──────────────────────────────────────────────────────────────────────────
**끝난 줄기: 평가함수 개선.** V12b 는 최종 잔여 사과를 평균 0.81개 오차로 맞히는
매우 정확한 가치를 갖고도, 조잡한 휴리스틱을 쓰는 V1.0 과 **통계적으로 같은
점수**를 냈다 (−0.19 ± 0.86). V6 부터 V12 까지 전부 이 줄기였다.

**열린 줄기: 평가의 편향.** 같은 시드 20판에서:

    min-area (base 정책)                        ~107
    1수앞 휴리스틱                               118.20
    빔 깊이 2 / 3 (평가 고정)                    119.20 / 119.00
    **rollout-1ply (후보마다 끝까지 두어 봄)      122.90**

깊이를 늘려도 평가가 나쁘면 +1 에서 멈춘다. 반면 평가를 **정확하게**(몬테카를로
롤아웃) 하면 +4.7 이고, base 정책(107) 기준으로는 **+16** 이다.

그리고 그 롤아웃은 K=1 이라 추정 잡음이 std ~5 로 V12b 의 MAE 0.81 보다 6배
크다. **그런데도 이긴다.** 문제는 정밀도가 아니라 **편향**이다.

**편향의 정체.** V^π 는 자기 정책이 지나간 상태에서만 학습된다. ε=0.05 면
데이터의 95% 가 탐욕 경로인데, 개선 연산자(형제 28개 중 최선 고르기)가 묻는
27개는 분포 밖이다. 가치가 자기 선택을 계속 1등으로 매기면 정책은 영원히
안 바뀐다 — **자기확인 고리**다. 롤아웃은 그 형제를 **실제로 두어 보므로**
고리를 끊는다.

(알고리즘 팀의 어닐링이 "최소 사과 제거" 라는 사소한 기준 하나로 ~137 을 내는
것도 같은 이유다. 기준이 좋아서가 아니라, 완성된 수순을 **실제 점수로** 평가하기
때문이다. 근사 오차가 0 이다.)

──────────────────────────────────────────────────────────────────────────
구조
──────────────────────────────────────────────────────────────────────────
    GridEncoder (V1.0/V12 와 같은 트렁크)
      ├─ RectangleHead   7533개 로짓, 신경망 안에서 마스킹  <- 롤아웃 시뮬레이터이자 경량 모델
      └─ ValueHead       예상 남은 점수 (보조/진단용)

`forward()` 가 두 모드를 갖는다. **가중치는 완전히 같고 상수만 다르다**
(V9c/V9cB 와 같은 방식이라 같은 체크포인트가 양쪽 폴더에서 로드된다).

    ROLLOUT_TOPK = 0   경량 — 정책 로짓의 argmax. 1회 forward.
    ROLLOUT_TOPK > 0   무거움 — 후보 K개를 끝까지 두어 보고 최종 점수 최고를 선택.

롤아웃은 전부 GPU 텐서 연산이고 **배치로 lockstep** 으로 돈다 (후보 16개면
한 번에 16판을 동시에 진행). 게임 엔진을 부르지 않는다.

**정책이 결정적이면 롤아웃 한 번이 곧 정확한 V^π 다** — 잡음이 0 이라
ROLLOUT_SAMPLES=1 로 충분하다. 확률적으로 굴리고 싶으면 온도를 올리고 표본을 늘린다.
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
VALUE_HIDDEN = 256
N_GROUPS = 8
MAX_APPLE_COUNT = 10
TARGET_SUM = 10

MASK_FILL = -1e8      # 불법 수
PRUNED_FILL = -1.0    # 합법이지만 후보에서 빠진 수 (실제 점수는 항상 0 이상이라 절대 안 뽑힌다)

N_DIGITS = 10
DENSITY_LOG_SCALE = math.log1p(64.0)   # env.py 와 반드시 같아야 한다
VALID_ACTION_SCALE = 128.0             # env.py 와 반드시 같아야 한다

# ── 롤아웃 설정 (이 상수만 바꾼 쌍둥이 폴더가 경량/무거움을 나눈다) ──────────
ROLLOUT_TOPK = 16        # 0 이면 롤아웃 없이 정책 로짓 argmax (경량 모델)
ROLLOUT_SAMPLES = 1      # 결정적 롤아웃이면 1 이 곧 정확한 V^pi 다
ROLLOUT_TEMPERATURE = 0.0  # 0 이면 argmax. >0 이면 확률적 롤아웃(표본을 늘려 평균)


def prefix_sum(plane: torch.Tensor) -> torch.Tensor:
    """(..., R, C) -> (..., R+1, C+1) 0 패딩 2차원 누적합."""
    return F.pad(plane, (1, 0, 1, 0)).cumsum(dim=-2).cumsum(dim=-1)


def grid_from_obs(observations: torch.Tensor) -> torch.Tensor:
    """관측의 숫자 원-핫 10채널에서 숫자판을 복원한다. (N, R, C)"""
    digits = torch.arange(N_DIGITS, device=observations.device,
                          dtype=observations.dtype)[None, :, None, None]
    return (observations[:, :N_DIGITS] * digits).sum(dim=1)


class RectIndex(nn.Module):
    """행동 <-> 직사각형. 합법 판정, 사과 수, 판 지우기, 관측 인코딩.

    전부 `game.Board` / `env._get_obs()` 와 **정확히 같아야** 한다. 한 곳이라도
    어긋나면 롤아웃이 실제 게임과 다른 게임을 두게 되는데, 아무 에러도 안 나고
    점수만 조용히 깎인다. `ai/verify/verify_v13.py` 가 매번 대조한다.
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
        """(M, R, C) -> (M, 7533) bool. game.Board 와 같은 판정."""
        prefix = prefix_sum(grids)
        total = self.total(prefix)
        top = self._area(prefix, self.r_lo, self.c_lo, self.r_lo + 1, self.c_hi)
        bottom = self._area(prefix, self.r_hi - 1, self.c_lo, self.r_hi, self.c_hi)
        left = self._area(prefix, self.r_lo, self.c_lo, self.r_hi, self.c_lo + 1)
        right = self._area(prefix, self.r_lo, self.c_hi - 1, self.r_hi, self.c_hi)
        return ((total - TARGET_SUM).abs() < 0.5) & (top > 0) & (bottom > 0) & (left > 0) & (right > 0)

    def apple_counts(self, grids: torch.Tensor) -> torch.Tensor:
        """(M, R, C) -> (M, 7533) 그 사각형 안의 사과 개수."""
        return self.total(prefix_sum((grids != 0).to(grids.dtype)))

    def erase(self, grids: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """(M, R, C) 판마다 action_idx[i] 사각형을 0 으로."""
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
    """관측 -> 셀별 임베딩. V12 와 같다."""

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


class RectangleHead(nn.Module):
    """셀 임베딩에서 7533개 직사각형의 로짓을 한 번에. V1.0 의 헤드 그대로다.

    V13 에서 이 헤드의 역할이 지금까지와 다르다. 예전에는 이것이 **가치**(Q)였고
    TD 로 배웠다. 여기서는 **정책**이고 롤아웃이 고른 수를 margin 손실로 배운다.
    가치 추정을 전혀 안 하므로 형제 서열 문제(V7 이후의 벽)를 아예 우회한다.
    """

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
        flat = cells.flatten(2).transpose(1, 2)                       # (N, R*C, E)
        pair = torch.bmm(self.proj_tl(flat),
                         self.proj_br(flat).transpose(1, 2)) * self.scale
        logits = pair[:, self.index.tl_idx, self.index.br_idx]
        logits = logits + self.index.total(prefix_sum(self.region(cells).squeeze(1)))

        count = self.index.total(occ_prefix).round().long().clamp_(0, MAX_APPLE_COUNT)
        return logits + self.shape_count_bias.view(-1)[
            self.index.shape_idx * self.n_counts + count]


class ValueHead(nn.Module):
    """예상 남은 점수 / 162. 보조·진단용이고 행동 선택에는 쓰이지 않는다.

    두는 이유: (1) 트렁크에 몬테카를로 감독이 하나 더 붙고, (2) `train/value_mae`
    로 '가치가 얼마나 정확한가' 를 V12b(0.808)와 같은 자로 볼 수 있다.
    행동 선택에 쓰지 않는 이유는 그것이 바로 V12 에서 막힌 길이기 때문이다.
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
        flat = cells.flatten(2)
        remaining = occupied.flatten(1).sum(dim=1, keepdim=True) / self.n_cells
        pooled = torch.cat([flat.mean(dim=2), flat.amax(dim=2), remaining], dim=1)
        return self.mlp(pooled).squeeze(1)


class RolloutQNetwork(QNetwork):
    """행동 선택을 **롤아웃**으로 한다. 가치를 추정하지 않고 실제로 두어 본다."""

    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        super().__init__(*args, **kwargs)
        assert actions is not None, "policy_kwargs에 actions=get_all_action(rows, cols)가 필요합니다."

        fe = self.features_extractor
        self.rows, self.cols = fe.rows, fe.cols          # type: ignore[assignment]
        self.embed_dim = fe.embed_dim                    # type: ignore[assignment]
        self.n_cells = self.rows * self.cols
        self.index = RectIndex(actions, self.rows, self.cols)
        self.q_net = RectangleHead(self.rows, self.cols, self.embed_dim,
                                   self.index, head_dim)   # 이름은 SB3 관례
        self.value_net = ValueHead(self.embed_dim, self.n_cells)

    # ── 기본 경로 ────────────────────────────────────────────────────────
    def _cells(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return features.view(features.shape[0], self.embed_dim, self.rows, self.cols)

    def policy_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """(N, 7533) 마스킹된 정책 로짓. 학습과 롤아웃이 쓰는 경로."""
        cells = self._cells(obs)
        occupied = (obs[:, 0] < 0.5).to(obs.dtype)
        logits = self.q_net(cells, prefix_sum(occupied))
        mask = self.index.legal_mask_from_grid(grid_from_obs(obs))
        return logits.masked_fill(~mask, MASK_FILL)

    def predict_value(self, obs: torch.Tensor) -> torch.Tensor:
        """예상 남은 점수 / 162. (N,)"""
        return self.value_net(self._cells(obs), obs[:, 0] < 0.5)

    def legal_mask(self, obs) -> torch.Tensor:
        return self.index.legal_mask_from_grid(grid_from_obs(obs))

    # ── 롤아웃 ───────────────────────────────────────────────────────────
    def _logits_from_grid(self, grids: torch.Tensor) -> torch.Tensor:
        return self.policy_logits(self.index.observation(grids))

    @torch.no_grad()
    def rollout_returns(self, grids: torch.Tensor, state_idx: torch.Tensor,
                        action_idx: torch.Tensor, temperature: float) -> torch.Tensor:
        """각 (상태, 후보수) 에 대해 **그 수를 두고 정책으로 끝까지** 둔 총 사과 수.

        후보 전부를 한 배치로 lockstep 진행한다. 판이 끝난 것은 빠지고 남은 것만
        계속 돈다. 합법수는 반드시 사과를 1개 이상 지우므로 칸 수만큼 돌면 반드시 끝난다.
        """
        boards = self.index.erase(grids[state_idx], action_idx)
        total = self.index.apple_counts(grids)[state_idx, action_idx].clone()

        for _ in range(self.n_cells):
            mask = self.index.legal_mask_from_grid(boards)
            alive = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if alive.numel() == 0:
                break
            live_boards = boards[alive]
            logits = self._logits_from_grid(live_boards)
            if temperature <= 0:
                action = logits.argmax(dim=1)
            else:
                action = torch.multinomial(
                    torch.softmax(logits / temperature, dim=1), 1).squeeze(1)
            total[alive] += self.index.apple_counts(live_boards).gather(
                1, action[:, None]).squeeze(1)
            boards[alive] = self.index.erase(live_boards, action)
        return total

    def forward(self, obs) -> torch.Tensor:
        """(N, 7533). ROLLOUT_TOPK 가 0 이면 정책 로짓, 아니면 롤아웃 점수."""
        logits = self.policy_logits(obs)
        if ROLLOUT_TOPK <= 0:
            return logits

        grids = grid_from_obs(obs)
        mask = self.index.legal_mask_from_grid(grids)
        n_legal = int(mask.sum(dim=1).max().item())
        if n_legal == 0:
            return logits

        top_k = min(ROLLOUT_TOPK, n_legal)
        candidates = logits.topk(top_k, dim=1).indices                    # (N, k)
        keep = mask.gather(1, candidates)          # 합법수가 k 보다 적은 상태가 있다
        rows = torch.arange(obs.shape[0], device=obs.device)[:, None].expand_as(candidates)
        state_idx, action_idx = rows[keep], candidates[keep]

        # 표본을 늘리려면 같은 후보를 여러 번 굴려 평균낸다 (결정적이면 1로 충분)
        repeat = max(int(ROLLOUT_SAMPLES), 1)
        returns = self.rollout_returns(
            grids, state_idx.repeat(repeat), action_idx.repeat(repeat), ROLLOUT_TEMPERATURE)
        returns = returns.view(repeat, -1).mean(dim=0)

        out = torch.full_like(logits, MASK_FILL)
        out = out.masked_fill(mask, PRUNED_FILL)   # 합법이지만 후보에서 빠진 수
        out[state_idx, action_idx] = returns / self.n_cells
        return out


class RolloutPolicy(DQNPolicy):
    def __init__(self, *args, actions: list[Action] | None = None,
                 head_dim: int = HEAD_DIM, **kwargs):
        self._actions = actions
        self._head_dim = head_dim
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> RolloutQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return RolloutQNetwork(**net_args, actions=self._actions,
                               head_dim=self._head_dim).to(self.device)


POLICY_CLASS = RolloutPolicy


def make_policy_kwargs(rows: int, cols: int) -> dict:
    return dict(
        features_extractor_class=GridEncoder,
        features_extractor_kwargs=dict(width=WIDTH, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM),
        net_arch=[],
        normalize_images=False,
        actions=get_all_action(rows, cols),
        head_dim=HEAD_DIM,
    )
