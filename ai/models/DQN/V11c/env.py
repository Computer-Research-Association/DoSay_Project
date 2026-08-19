"""DQN V11c 환경 — 관측은 V1.0 과 같다. **퍼텐셜 셰이핑을 뺐고, 최종 판을 info 로 내보낸다.**

──────────────────────────────────────────────────────────────────────────
1. 왜 셰이핑(Φ)을 뺐는가
──────────────────────────────────────────────────────────────────────────
Φ = 0.3·log1p(합법수) 는 '남는 선택지가 많은 쪽이 좋다'는 정보를 보상에 넣어
형제 서열 신호를 보태려던 장치다. 그런데 **Q 학습에서 이것은 행동 선택을
전혀 바꾸지 못한다.** 셰이핑을 쓰면 신경망이 배우는 것은

    Q_shaped(s,a) = Q_true(s,a) - Φ(s)

인데 Φ(s) 는 a 에 무관한 상태 상수라 argmax 가 그대로다 (V1.0 env.py 도 이 점을
근거로 셰이핑을 넣었다 — "argmax 가 그대로라 안전하다"). 즉 DQN 계열에서 Φ 는
**목표 스케일만 흔들고 정책에는 기여하지 않았다.**

V11c 는 그 자리에 진짜 행동별 항을 넣는다 (model.py 의 look 항). 손으로 고른
계수 0.3 을 지우고, 신경망이 그 항의 값과 가중치를 **스스로 배우게** 한다.
둘을 겹쳐 두면 무엇이 효과를 냈는지 알 수 없으므로 Φ 는 뺀다.

(참고: 셰이핑이 없으므로 에피소드 보상 합 = 점수/162 다. `rollout/ep_rew_mean`
을 162배 하면 그대로 점수라 로그를 읽기도 쉬워진다.)

──────────────────────────────────────────────────────────────────────────
2. 왜 최종 판(final_grid)을 info 로 내보내는가
──────────────────────────────────────────────────────────────────────────
V11c 의 보조 목표 중 하나가 **"이 사과가 끝까지 남을 확률"** 을 칸마다 예측하는
것이다. 그 정답은 에피소드가 끝나 봐야 알 수 있으므로, 종료 시점의 판을
그대로 내보내 학습 루프가 되짚어 라벨을 채우게 한다.

이 라벨이 필요한 이유는 게임의 회계 구조에 있다. 한 수는 **항상 합 10** 을
가져가고 판 합은 810 으로 고정이므로

    수 = 제거한 합 / 10,   점수 = 제거한 사과 수,   점수 = 162 - 남은 사과 수

이다. 그리고 초기 재고는 각 숫자 18개씩이라 9+1, 8+2, 7+3, 6+4, 5+5 로 짝지으면
정확히 81수 x 2칸 = 162칸, 합 810 — **완전 클리어와 딱 맞아떨어진다.** 그래서
`1+2+3+4=10` 처럼 한 수에 4개를 먹는 것은 당장 4점이지만 9·8·7·6 의 짝을
동시에 태워 4점을 영구히 잃는다(짝지었으면 8점이다). 문서에 남아 있는
"최다 제거 탐욕 91점 < 무작위 96.2점" 이 여기서 설명된다.

실측한 잔여 구성도 그대로다 (min-area 정책, 초기 각 18개 기준 남은 비율):

    1: 14%   2: 19%   3: 21%   4: 23%   5: 38%   6: 51%   7: 46%   8: 50%   9: 52%

작은 숫자를 다 태우고 큰 숫자를 좌초시킨다. 그리고 5개 모델을 점/수로 줄 세우면
2.0(완전 짝짓기)에 가까울수록 점수가 높다.

    점/수  2.224 -> 121.45,  2.240 -> 114.48,  2.246 -> 120.62,
           2.263 -> 111.08,  2.270 -> 107.65

**'어떤 사과가 좌초되는가'가 이 게임의 핵심 상태변수인데 지금까지 어떤 학습
신호도 그것을 직접 가리키지 않았다.** 잔존맵은 그 구조를 신경망이 표현하도록
만드는 장치이고, 라벨은 에이전트 자신의 플레이에서 나오므로 손으로 짠 규칙이
하나도 들어가지 않는다.

**평가는 영향받지 않는다.** info 에 키가 하나 늘 뿐 게임 규칙과 점수는 그대로다.
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
SEARCH_TOP_K = 0    # 순수 신경망. 추론은 1회 forward.
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
        """셰이핑 없음. 에피소드 보상 합이 정확히 점수/162 다."""
        return (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD

    def _get_info(self):
        info = super()._get_info()
        # 종료 상태에서만 최종 판을 싣는다. 매 스텝 실으면 162바이트씩 프로세스 간
        # 직렬화 비용이 붙는데, 필요한 것은 마지막 한 번뿐이다.
        # (SubprocVecEnv 는 done 인 스텝의 info 를 그대로 학습 프로세스로 넘긴다.)
        if not self.board.get_valid_actions():
            info["final_grid"] = self.board.grid.copy()
        return info


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """평가·학습 공용. V11c 는 학습용 판 공급을 바꾸지 않는다 —
    이 버전이 검증하려는 것은 '감독 신호의 밀도'이지 판 공급 방식이 아니다.
    (판 공급을 바꾸는 실험은 V11a / V11b 가 맡는다.)
    """
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)
