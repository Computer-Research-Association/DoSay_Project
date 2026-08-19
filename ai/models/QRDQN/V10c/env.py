"""QR-DQN V10c 환경 — V9a 와 동일. 탐색은 **꺼져 있다**.

V10c 는 학습 중에만 탐색을 교사로 쓰고, 배포 모델은 신경망 argmax 만으로 둔다.
그래서 이 파일의 SEARCH_TOP_K 는 0 이어야 한다. 탐색 설정은 train.py 안에 있다.
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

# ── 로드할 때 건너뛸 학습 전용 상태 (runs/agents/ai/model_loader.py 가 읽는다) ──
#
# ExpertIterationQRDQN 은 탐색 교사용 env(`_search_env`)와 교사 데이터
# (`expert_obs`/`expert_actions`)를 인스턴스 속성으로 들고 있다. SB3 는
# 알고리즘의 __dict__ 를 통째로 저장하므로 이것들이 체크포인트에 함께 들어갔다.
#
# `_search_env` 는 이 파일의 AppleGameEnv 인스턴스다. 학습 중에는 이 파일이
# 최상위 모듈 `env` 로 import 돼 있어서 cloudpickle 이 "모듈 env 의 클래스"라는
# **참조**로 저장한다. measure.py 에는 `env` 라는 모듈이 없으므로 로드가
# ModuleNotFoundError: No module named 'env' 로 죽었다.
#
# 셋 다 추론에는 전혀 쓰이지 않는다(배포 모델은 신경망 argmax 만 쓴다). 그래서
# 역직렬화를 건너뛰고 빈 값으로 대체한다 — **재학습 없이** 기존 체크포인트가
# 그대로 로드된다. 새로 저장되는 체크포인트는 train.py 의
# _excluded_save_params() 가 애초에 이것들을 빼므로 이 항목이 필요 없다.
LOAD_CUSTOM_OBJECTS = {
    "_search_env": None,
    "expert_obs": [],
    "expert_actions": [],
}


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
