"""MaskablePPO V9c 환경 — 측정으로 이긴 것만 모으고 후반 커리큘럼을 얹는다.

V8 100판 결과(같은 seed 짝지어 비교):

    전용 critic(V8c vs V8b)   +5.38 ± 1.28   <- 가장 큰 승리
    셰이핑 0.3 vs 1.0(V8a/V8b) +2.45 ± 1.40
    빔탐색(V8a) vs DQN(탐색없음) +0.46 ± 1.39  <- 사실상 0

그래서 신경망은 V8c(전용 critic)를 그대로 쓰고 셰이핑만 0.3 으로 되돌린다.
V8c 는 1.0 이었으므로 이 둘을 합친 조합은 아직 돌려본 적이 없다.

여기에 후반 커리큘럼을 더한다. 모든 모델이 사과를 46~51개 남기고 끝나는데,
후반 상태는 에피소드 꼬리에만 나와 경사의 20%도 못 받는다. 학습할 때만
에피소드 절반을 이미 진행된 판에서 시작해 후반 연습량을 늘린다.
평가는 make_env() 를 쓰므로 언제나 꽉 찬 판에서 시작한다.

V9cB 는 V9c 와 완전히 같은 조합에 빔탐색만 켠 것이다.
둘을 비교하면 더 강해진 모델 위에서 탐색이 실제로 얼마를 더하는지 나온다.
"""

import math

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ai.envs.action_sets import get_all_action
from ai.envs.base_env import AppleGameEnvBase
from game.action import Action
from game.board import Board

N_DIGITS = 10                      # 0(빈칸) ~ 9
DENSITY_LOG_SCALE = float(np.log1p(64.0))
VALID_ACTION_SCALE = 128.0

# 형제 간 Φ 차이가 V 의 잔차(0.0103)보다 커야 서열을 주도한다.
# 합법수가 50 -> 45 로 줄 때 Δlog1p ≈ 0.103 이므로 1.0 이면 ΔΦ ≈ 0.103 (잔차의 10배).
POTENTIAL_SCALE = 0.3

# ── 탐색 설정 (runs/agents/ai/search.py 가 읽는다) ────────────────────────
SEARCH_TOP_K = 64
SEARCH_DEPTH = 2
SEARCH_BEAM_WIDTH = 6
SEARCH_BEAM_TOP_K = 16


class AppleGameEnv(AppleGameEnvBase):
    STEP_REWARD_TOTAL = 1.0
    INVALID_ACTION_REWARD = -1.0

    N_CHANNELS = N_DIGITS + 3

    def build_actions(self, rows: int, cols: int) -> list[Action]:
        return get_all_action(rows, cols)

    def build_observation_space(self, rows: int, cols: int) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(self.N_CHANNELS, rows, cols), dtype=np.float32)

    def _option_density(self) -> NDArray[np.float32]:
        """각 칸을 덮는 합법수의 개수. 2차원 차분 배열이라 합법수당 덧셈 4번이면 된다."""
        diff = np.zeros((self.row_num + 1, self.col_num + 1), dtype=np.int32)
        for action in self.board.get_valid_actions():
            (r1, c1), (r2, c2) = action.top_left, action.bottom_right
            diff[r1, c1] += 1
            diff[r1, c2 + 1] -= 1
            diff[r2 + 1, c1] -= 1
            diff[r2 + 1, c2 + 1] += 1

        density = diff.cumsum(axis=0).cumsum(axis=1)[:self.row_num, :self.col_num]
        return density.astype(np.float32)

    def _get_obs(self) -> NDArray[np.float32]:
        obs = np.zeros((self.N_CHANNELS, self.row_num, self.col_num), dtype=np.float32)

        # 0~9 : 숫자 원-핫. 0번 평면이 빈칸이라 모델이 점유 여부를 그대로 쓴다.
        digits = np.arange(N_DIGITS, dtype=np.int8)[:, None, None]
        obs[:N_DIGITS] = (self.board.grid[None, :, :] == digits)

        density = self._option_density()
        peak = float(density.max())
        obs[N_DIGITS] = density / peak if peak > 0 else 0.0
        obs[N_DIGITS + 1] = np.minimum(np.log1p(density) / DENSITY_LOG_SCALE, 1.0)
        obs[N_DIGITS + 2] = min(len(self.board.get_valid_actions()) / VALID_ACTION_SCALE, 1.0)

        return obs

    def potential(self) -> float:
        """Φ(s). 종료 상태(합법수 0)에서 정확히 0 이라 셰이핑 조건을 만족한다.

        탐색도 이 값을 쓴다 — V 는 V_true - Φ 를 학습하므로 Φ 를 되더해야 참값이다.
        """
        return POTENTIAL_SCALE * math.log1p(len(self.board.get_valid_actions()))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_potential = self.potential()
        return obs, info

    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        base = (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL

        potential = self.potential()          # 이미 수를 둔 뒤이므로 Φ(s')
        shaped = potential - self._prev_potential
        self._prev_potential = potential

        return base + shaped

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD


CURRICULUM_RATIO = 0.5
CURRICULUM_MAX_PREFIX = 35


class CurriculumEnv(AppleGameEnv):
    """학습 전용. 일정 비율의 에피소드를 이미 진행된 판에서 시작한다."""

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.np_random.random() >= CURRICULUM_RATIO:
            return obs, info

        prefix = int(self.np_random.integers(1, CURRICULUM_MAX_PREFIX + 1))
        safe = self.board.grid.copy()
        for _ in range(prefix):
            actions = self.board.get_valid_actions()
            if not actions:
                break
            index = int(self.np_random.integers(len(actions)))
            _, removed = self.board.do_action(actions[index])
            self.score += removed

            if not self.board.get_valid_actions():
                # 이 수로 판이 죽었다. 그대로 두면 '시작하자마자 둘 수 없는 판' 이 되는데,
                # base_env.step 은 둘 수 없는 수에 terminated=False 를 돌려주므로
                # 그 환경의 에피소드가 영원히 끝나지 않는다. 직전 상태로 되돌린다.
                self.board = Board.from_board(safe)
                self.score -= removed
                break
            safe = self.board.grid.copy()

        self._prev_potential = self.potential()
        return self._get_obs(), self._get_info()


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """평가용. 언제나 꽉 찬 판에서 시작한다."""
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)


def make_train_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """학습용. 후반 커리큘럼이 걸려 있다."""
    return CurriculumEnv(rows=rows, cols=cols, render_mode=render_mode)
