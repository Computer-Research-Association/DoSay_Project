"""DQN V14 환경 — 관측은 V1.0/V12/V13 과 같다. **보상은 학습에 쓰이지 않는다.**

V14 의 학습 신호는 둘이고 둘 다 보상이 아니다.

    정책 헤드:  빔이 고른 수 (margin 손실)
    가치 헤드:  그 판의 실제 최종 점수 (회귀)

관측이 V12b 와 **완전히 같아야 한다** — V14 는 V12b 체크포인트로 워밍스타트해서
1스텝째부터 강한 빔 교사를 쓰기 때문이다. 그리고 model.py 의
`RectIndex.observation()` 이 GPU 에서 이것과 원소 단위로 같은 것을 만들어야 한다
(빔이 매 수 그것을 쓴다). `ai/verify/verify_v14.py` 가 매 실행마다 대조한다.
"""

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ai.envs.action_sets import get_all_action
from ai.envs.base_env import AppleGameEnvBase
from game.action import Action

N_DIGITS = 10                      # 0(빈칸) ~ 9
DENSITY_LOG_SCALE = float(np.log1p(64.0))   # model.py 와 반드시 같아야 한다
VALID_ACTION_SCALE = 128.0                  # model.py 와 반드시 같아야 한다

# ── 탐색 설정 (runs/agents/ai/search.py 가 읽는다) ────────────────────────
# 끈다. V12 의 행동 선택 자체가 신경망 안에서 1수 앞을 정확히 계산하는 구조라,
# 그 위에 빔서치를 얹는 것은 별도 버전에서 따로 실험한다.
SEARCH_TOP_K = 0
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

    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        """학습에 쓰이지 않는다. 에피소드 합이 점수/162 라 로그 읽기에만 쓴다."""
        return (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD

    def _get_info(self):
        info = super()._get_info()
        # 종료 상태에서만 최종 판을 싣는다 (매 스텝 실으면 프로세스 간 직렬화 비용만 는다).
        if not self.board.get_valid_actions():
            info["final_grid"] = self.board.grid.copy()
        return info


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
