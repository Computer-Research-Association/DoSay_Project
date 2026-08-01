"""V7.0 환경 — 관측은 V6.0 과 같고, 대신 탐색을 켠다.

V6.0 학습 로그(100만 step)가 읽어 준 것:

    explained_variance  0.975      가치망은 남은 점수를 거의 정확히 맞힌다
    entropy_loss        -0.79      정책은 이미 거의 결정적인데도 105점
    ep_score_mean       30만 step에 102, 100만 step에 103

즉 병목은 '평가'가 아니라 '평가를 행동으로 옮기는 것'이다. 학습을 더 태워도
점수가 안 오른다. 반면 손으로 짠 1수 앞 탐색은 118점을 낸다.

그래서 V7.0 은 학습을 늘리는 대신 탐색을 켠다. 정확한 V 가 있으면 수를 실제로
두어 보고 고르면 되기 때문이다. runs/agents/ai/search.py 가 아래 상수를 읽는다.

보상은 remove_count/162 그대로 둔다. 퍼텐셜 셰이핑도 고려했지만 넣지 않았다.
셰이핑이 메워 줄 자리가 가치 추정인데 그게 이미 0.975 라 얻을 것이 없고,
대신 탐색 점수식에 보정항이 붙어 틀리기 쉬워진다. 지금 형태에서는
어떤 상태의 참값이 곧 '남은 점수/162' 라 경로 점수가 정확히 (먹은 합 + V) 다.
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

# ── 탐색 설정 (runs/agents/ai/search.py 가 읽는다) ────────────────────────
#
# 대역 V(log1p(남은 합법수))로 20판씩 재 본 결과:
#
#     탐색 없음                    101.0    0.2s/판
#     depth 1, top-64              116.7    7.1s/판
#     depth 2, beam 6, top-64/16   117.9   22.6s/판   (+1.2)
#     depth 3, beam 4, top-64/12   117.5   24.9s/판   (-0.5)
#
# 깊이 3은 오히려 나빴다. 대역 V 가 '선택지 개수'라는 근시안적 대용값이라,
# 깊게 팔수록 참목표가 아니라 그 대용값을 더 세게 최적화하기 때문이다.
# 진짜 V(V6.0 기준 explained_variance 0.975)는 최종 점수를 직접 예측하므로
# 깊이의 이득이 이 측정보다 클 가능성이 높다. 그래서 기본은 깊이 2로 둔다.
# 비용이 부담되면 SEARCH_DEPTH=1 이 3배 빠르고 위 실험 기준 1점 남짓 손해다.
#
# top_k=64 는 합법수가 보통 50개 안팎이라 사실상 루트 전수 탐색이다.
# 상위 24로 자르면 같은 조건에서 117.8 -> 114.7 로 3점을 잃었다.
SEARCH_TOP_K = 64
SEARCH_DEPTH = 2
SEARCH_BEAM_WIDTH = 6
SEARCH_BEAM_TOP_K = 16   # 2수째는 좁게. 한 수당 판 복제가 64 + 6x16 = 160회다.


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
        # 에피소드 보상 합 = 최종 점수 / 162. 탐색 점수식이 정확하려면 이대로여야 한다.
        return (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
