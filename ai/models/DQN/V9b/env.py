"""DQN V9b 환경 — 후반 커리큘럼. DQN V1.0 과 다른 점은 학습 시작 상태뿐이다.

**왜 후반인가.** 100판 벤치마크에서 모든 모델이 사과를 46~51개씩 남기고 끝난다.
즉 점수를 잃는 곳은 초반이 아니라 판이 말라가는 후반이다. 그런데 지금 학습에서
후반 상태는 에피소드 꼬리에만 나온다 — 한 판 50수 중 마지막 10수쯤이라
경사의 20%도 못 받는다. 정작 어려운 국면을 가장 적게 연습하는 셈이다.

그래서 학습할 때만 에피소드의 절반을 **이미 진행된 판**에서 시작한다. 판을
무작위 합법수로 k 수 진행시킨 뒤 거기서 학습을 시작하는 방식이라, 손으로 만든
후반 규칙이 아니라 실제로 도달 가능한 후반 분포를 그대로 쓴다.

평가에는 영향이 없다. measure.py 는 make_env() 를 쓰고 커리큘럼은
make_train_env() 에만 걸린다. 즉 채점은 언제나 꽉 찬 판에서 시작한다.
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
# 합법수가 50 -> 45 로 줄 때 Δlog1p ≈ 0.103 이므로 0.3 이면 ΔΦ ≈ 0.031 (잔차의 3배).
POTENTIAL_SCALE = 0.3

# ── 탐색 설정 (runs/agents/ai/search.py 가 읽는다) ────────────────────────
SEARCH_TOP_K = 0    # Q 의 argmax 자체가 1수 앞 탐색이다

# 학습 에피소드 중 이 비율만큼은 진행된 판에서 시작한다
CURRICULUM_RATIO = 0.5
CURRICULUM_MAX_PREFIX = 35   # 무작위로 진행시킬 최대 수
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
                # 이 수로 판이 죽었다. 그대로 두면 '시작하자마자 둘 수 없는 판' 이 되고,
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
