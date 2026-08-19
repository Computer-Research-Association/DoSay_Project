"""DQN V1.0 환경 — 관측과 보상은 V8a 와 같다. 알고리즘만 다르다.

이 버전의 질문은 하나다. **Q(s,a) 를 직접 배우면 V + 탐색보다 나은가?**

V7.0 이 드러낸 문제는 V 가 형제 상태를 서열 매기지 못한다는 것이었다
(EV 0.983 인데 잔차가 약 1.7점, 형제 간 가치 차이가 딱 그 정도). Q 는 애초에
(상태, 행동) 쌍마다 값을 갖기 때문에 형제 구분이 곧 학습 목표다. 즉 DQN 의
greedy argmax 는 그 자체로 '학습된 1수 앞 탐색' 이다. 그래서 이 버전은
빔 탐색을 켜지 않는다 (SEARCH_TOP_K = 0).

퍼텐셜 셰이핑은 V8a 와 같은 0.3 으로 맞췄다. 알고리즘 차이만 보기 위해서다.
DQN 에서 셰이핑이 행동 선택을 아예 바꾸지 않는다는 점도 좋다 —
Q_shaped(s,a) = Q_true(s,a) - Φ(s) 인데 Φ(s) 가 a 에 무관해 argmax 가 그대로다.

마스킹은 sb3 DQN 이 지원하지 않아 신경망 안에서 직접 만든다 (model.py 참고).
탐험(ε-greedy)도 합법수 중에서만 뽑도록 train.py 에서 고쳤다. 그대로 두면
7533개 중 균등추출이라 거의 전부 불법 수가 되어 학습이 성립하지 않는다.
"""

import math

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ai.envs.action_sets import get_all_action
from ai.envs.base_env import AppleGameEnvBase
from game.action import Action

N_DIGITS = 10                      # 0(빈칸) ~ 9
DENSITY_LOG_SCALE = float(np.log1p(64.0))
VALID_ACTION_SCALE = 128.0

# 형제 간 Φ 차이가 V 의 잔차(0.0103)보다 커야 서열을 주도한다.
# 합법수가 50 -> 45 로 줄 때 Δlog1p ≈ 0.103 이므로 0.3 이면 ΔΦ ≈ 0.031 (잔차의 3배).
POTENTIAL_SCALE = 0.3

# ── 탐색 설정 (runs/agents/ai/search.py 가 읽는다) ────────────────────────
SEARCH_TOP_K = 0    # Q 의 argmax 자체가 1수 앞 탐색이다
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


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
