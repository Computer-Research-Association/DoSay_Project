"""V8a 환경 — 퍼텐셜 셰이핑을 되살린다 (약하게, POTENTIAL_SCALE=0.3).

V7.0 이 왜 107.6 에 그쳤는지가 로그에 다 있다.

    explained_variance  0.983      전역 예측은 거의 완벽
    정책만의 점수       105.5
    탐색까지 붙여        107.6      (+2.1)
    V 자리에 log1p(합법수) 를 꽂으면  117.9   (+13)

조잡한 대용값이 잘 학습된 V 를 10점 차로 이긴다. EV 가 재는 것과 탐색이
필요로 하는 것이 다르기 때문이다.

    점수 표준편차 12.77 -> 보상 단위 0.0788
    EV 0.983 -> 잔차 표준편차 = 0.0788 * sqrt(0.017) = 0.0103 -> 약 1.7점

EV 는 '이 판이 대충 몇 점짜리인가'를 재는데 그건 남은 사과 수만 봐도 맞는다.
탐색이 필요로 하는 것은 **같은 판에서 갈라진 형제들의 서열**이고, 형제 간 가치
차이가 딱 그 잔차(1.7점)만 하다. 그래서 V 의 서열은 사실상 잡음이다.
반면 log1p(합법수) 는 절대값이 틀려도 서열이 일관돼서 이긴다.

퍼텐셜 셰이핑이 이 문제를 정면으로 푼다.

    Φ(s) = SCALE * log1p(남은 합법수),   F(s,s') = γΦ(s') - Φ(s)  (γ=1)

γ=1 이고 종료 상태는 합법수가 0 이라 Φ=0 이므로 Ng et al.(1999) 조건을 만족한다.
**최적 정책이 바뀌지 않는다** — 휴리스틱을 보상에 그냥 섞는 것과 다르다.
학습된 V 는 V_shaped = V_true - Φ 가 되므로, 탐색은 Φ 를 되더해 참값을 복원한다.

이 구조의 이점은 **바닥이 생긴다**는 것이다. V_shaped 가 잡음이면 탐색 점수가
(먹은 합 + Φ) 로 떨어져 118점짜리 대용값 동작이 되고, V_shaped 가 쓸모 있으면
그 위에 보정이 얹힌다. 즉 최악이 107 이 아니라 118 이다.

V8a 와 V8b 는 SCALE 만 다르다. V8a = 0.3 (V 주도, Φ 는 보조).
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


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
