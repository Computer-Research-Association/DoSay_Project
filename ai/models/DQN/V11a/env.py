"""DQN V11a 환경 — 관측·보상은 V1.0 과 완전히 같다. **학습용 판 공급 방식만** 다르다.

V1.0(114.48점)을 기준선으로 삼고 두 가지를 바꾼 것이 V11a 인데, 그중 이 파일이
담당하는 것은 **같은 판을 GROUP_SIZE 번 연속으로 둔다**는 것 하나다.

**왜.** V10b 의 자기모방이 실패한 이유를 100판 벤치마크로 되짚어 보면 이렇다.
5개 모델을 같은 시드 100판으로 이원배치 분산분해하면

    판(시드) 성분  172.9 (76.1%)   <- 압도적
    모델 성분       28.6 (12.6%)
    상호작용/잔차   25.7 (11.3%)

이고, 어떤 모델이든 자기 상위 20% 에피소드를 골라 **다른 모델**의 같은 시드 점수를
보면 그쪽도 평균보다 +14~16점 높다. 즉 V10b 의 EliteBuffer 가 걸러낸 것은
'잘 둔 판'이 아니라 '쉬운 판'이었다. 신호는 거의 없고 전부 판 운이었다.
`train/elite_threshold` 가 107 -> 117 로 예쁘게 올라간 것도 "쉬운 판을 점점 더
잘 골라냈다"는 뜻일 뿐이다.

**판을 고정하면 이 성분이 통계적으로 줄어드는 게 아니라 정확히 상쇄된다.**
같은 판을 K번 두고 그 K개끼리만 비교하면 남는 차이는 정책의 차이뿐이다.
실측한 크기도 충분하다 — 무작위 타이브레이크를 준 min-area 정책으로 12판 x 12회를
재보면 판 간 std 12.73, **판 안 std 5.41**, 같은 판 best-of-12 가 평균보다 **+8.62**다.
한심한 휴리스틱조차 같은 판 재시도만으로 8.6점이 나온다. 그게 지금 우리 학습 신호가
판 운에 파묻혀 통째로 못 보고 있던 축이다.

**평가는 영향받지 않는다.** measure.py 는 make_env() 를 쓰고 이 파일의 그룹 로직은
make_train_env() 에만 들어 있다 (V9b 커리큘럼과 같은 규약).

퍼텐셜 셰이핑 Φ=0.3 은 V1.0 그대로 둔다. 알고리즘 축을 하나라도 더 흔들지 않기 위해서다.
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

# 형제 간 Φ 차이가 V 의 잔차(0.0103)보다 커야 서열을 주도한다. V1.0/V8a 와 같은 값.
POTENTIAL_SCALE = 0.3

# 한 판을 몇 번 반복해서 둘 것인가. 크면 선택이 정확해지고(best-of-K), 작으면
# 같은 시간에 더 다양한 판을 본다. 8 이면 6M 스텝에 약 15,000개의 서로 다른 판을 본다.
GROUP_SIZE = 8

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

    def potential(self) -> float:
        """Φ(s). 종료 상태(합법수 0)에서 정확히 0 이라 셰이핑 조건을 만족한다."""
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


class GroupBoardEnv(AppleGameEnv):
    """같은 판을 repeat 번 연속으로 둔다. **학습 전용**이다.

    board_source 를 명시적으로 주면(평가·디버깅) 그 판을 그대로 쓰고 그룹 로직은
    비켜간다. 그래서 이 클래스를 써도 seed 를 지정한 재현 실행은 그대로 된다.

    info 로 내보내는 것:
        group_index   이 에피소드가 속한 그룹 번호 (환경마다 독립적으로 증가)
        group_slot    그룹 안에서 몇 번째 판인가 (1..repeat)

    train.py 는 group_index 가 바뀌는 순간을 그룹 마감으로 보고, 그 그룹 안에서
    가장 점수가 높았던 롤아웃만 모방 대상으로 채택한다.
    """

    def __init__(self, *args, repeat: int = GROUP_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self._repeat = repeat
        self._left = 0
        self._board_seed: int | None = None
        self._group_index = -1
        self._group_slot = 0
        # 환경마다(=프로세스마다) 서로 다른 판을 보도록 OS 엔트로피로 초기화한다.
        self._board_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if options.get("board_source") is None:
            if self._left <= 0:
                self._board_seed = int(self._board_rng.integers(0, 2 ** 31 - 1))
                self._left = self._repeat
                self._group_index += 1
                self._group_slot = 0
            options["board_source"] = self._board_seed
            self._left -= 1
            self._group_slot += 1
        return super().reset(seed=seed, options=options)

    def _get_info(self):
        info = super()._get_info()
        info["group_index"] = self._group_index
        info["group_slot"] = self._group_slot
        return info


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """평가·측정용. 매 판 새로운 무작위 판에서 시작한다."""
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)


def make_train_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """학습용. 같은 판을 GROUP_SIZE 번 반복한다."""
    return GroupBoardEnv(rows=rows, cols=cols, render_mode=render_mode, repeat=GROUP_SIZE)
