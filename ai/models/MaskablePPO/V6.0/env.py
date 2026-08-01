"""V6.0 환경 — 관측에 '옵션 밀도'를 넣는다.

측정해 보면 이 게임은 많이 먹는 게임이 아니다. (seed 1234~, 만점 162)

    최다 제거 탐욕   91.0
    무작위 합법수    96.2
    작은 넓이 우선  108.5
    1수 앞 유연성   118.1   <- 둔 뒤 '남는 합법수'가 가장 많은 수를 고르는 한 줄 휴리스틱

즉 한 수의 가치는 먹는 사과 수가 아니라 **그 수가 파괴하는 미래의 선택지**로
결정된다. 그래서 숫자판만 주는 대신, 각 칸을 덮는 합법수의 개수(옵션 밀도)를
함께 준다. 정책 헤드는 직사각형 내부의 밀도 합을 O(1) 에 읽어
"이 수가 얼마나 많은 선택지를 부수는가"를 바로 알 수 있다.

보상은 손대지 않는다. remove_count / 162 의 합은 정확히 최종 점수 / 162 라서
학습이 최대화하는 값과 우리가 재는 값이 같다. 여기에 휴리스틱을 섞으면
그 휴리스틱의 상한에 갇힌다.
"""

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ai.envs.action_sets import get_all_action
from ai.envs.base_env import AppleGameEnvBase
from game.action import Action

N_DIGITS = 10                      # 0(빈칸) ~ 9
DENSITY_LOG_SCALE = float(np.log1p(64.0))
VALID_ACTION_SCALE = 128.0


class AppleGameEnv(AppleGameEnvBase):
    STEP_REWARD_TOTAL = 1.0
    INVALID_ACTION_REWARD = -1.0

    N_CHANNELS = N_DIGITS + 3

    def build_actions(self, rows: int, cols: int) -> list[Action]:
        return get_all_action(rows, cols)

    def build_observation_space(self, rows: int, cols: int) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(self.N_CHANNELS, rows, cols), dtype=np.float32)

    def _option_density(self) -> NDArray[np.float32]:
        """각 칸을 덮는 합법수의 개수.

        2차원 차분 배열이라 합법수 하나당 4번의 덧셈이면 된다. 합법수가 보통
        수십 개뿐이라 매 스텝 계산해도 부담이 없다.
        """
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
        # 상대 밀도: 이번 판에서 어디가 요충지인가
        obs[N_DIGITS] = density / peak if peak > 0 else 0.0
        # 절대 밀도: 요충지가 얼마나 두꺼운가 (판이 풀렸는지 말랐는지)
        obs[N_DIGITS + 1] = np.minimum(np.log1p(density) / DENSITY_LOG_SCALE, 1.0)
        # 남은 선택지 총량 (전 칸 동일값) — 종료가 얼마나 가까운지
        obs[N_DIGITS + 2] = min(len(self.board.get_valid_actions()) / VALID_ACTION_SCALE, 1.0)

        return obs

    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        # 에피소드 보상 합 = 최종 점수 / 162. 별도의 종료 보너스를 두지 않는다.
        return (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
