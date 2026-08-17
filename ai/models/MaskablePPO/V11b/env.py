"""MaskablePPO V11b 환경 — **보상을 판 단위 상대값으로 바꾼다.** (그룹 상대 정책경사)

관측 인코딩은 V9c 와 한 글자도 다르지 않다. 바뀐 것은 보상 하나다.

    (지금까지)  매 수마다  먹은사과/162  + [Φ(s') - Φ(s)]
    (V11b)      매 수마다  0
                종료할 때  (이 판 점수 - 같은 판의 다른 롤아웃 평균) / GROUP_SCORE_SCALE

──────────────────────────────────────────────────────────────────────────
왜 이렇게 하는가
──────────────────────────────────────────────────────────────────────────
5개 모델(V1.0 / V10a / V10b / V9cB / V9dB)을 같은 시드 100판으로 이원배치
분산분해하면 이렇게 나온다.

    판(시드) 성분   172.9   (76.1%)   std 13.15
    모델 성분        28.6   (12.6%)   std  5.34
    상호작용/잔차    25.7   (11.3%)   std  5.07

**점수 분산의 76%가 어떤 판을 뽑았느냐**다. 모델 간 상관도 r=0.77~0.90 으로,
같은 판에서는 어느 모델이든 비슷한 점수를 낸다. 지금까지의 학습 신호는 전부
이 잡음 위에 얹혀 있었다.

**같은 판을 K번 두고 그 K개끼리만 비교하면 판 성분은 통계적으로 줄어드는 게
아니라 정확히 상쇄된다.** 남는 것이 정책의 차이다. 그리고 그 차이는 작지 않다 —
무작위 타이브레이크를 준 min-area 정책으로 12판 x 12회를 재보면

    판 간 std        12.73
    판 안 std         5.41      <- 이것이 그룹 베이스라인이 쓰는 신호
    같은 판 best-of-12  평균 +8.62

한심한 휴리스틱조차 같은 판 재시도만으로 8.6점이 나온다. 알고리즘 팀의 어닐링이
추론 시점에 하는 것이 바로 이 재시도인데, 우리는 그것을 **학습 시점에 흡수해서
추론은 1회 forward 로 끝낸다.** 그게 이 버전의 요지다.

──────────────────────────────────────────────────────────────────────────
설계 결정 세 가지
──────────────────────────────────────────────────────────────────────────
1) **베이스라인은 '이 판에서 이미 끝난 롤아웃들의 평균'이다.** 현재 에피소드의
   결과에 의존하지 않으므로 통제변량으로서 편향이 없다. 그룹의 첫 롤아웃은
   비교 대상이 없는데, 이때 보상을 0 으로 두면 A_t = -V(s_t) 라는 **가짜 경사**가
   생긴다. 그래서 첫 롤아웃은 이 환경이 지금까지 본 전체 평균(EMA)을 쓴다 —
   판 운이 안 빠질 뿐 여전히 유효한 베이스라인이다.

2) **퍼텐셜 셰이핑을 뺐다.** Φ 는 형제 서열 신호를 보태려고 넣은 장치인데,
   그룹 상대 보상은 그 신호를 훨씬 직접적으로 만들어 준다. 둘을 겹치면 무엇이
   효과를 냈는지 알 수 없다. 그리고 셰이핑이 없으면 에피소드 보상 합이 곧
   (점수 - 베이스라인) 이라 해석이 깨끗하다.

3) **매 수 보상을 0 으로 둔 이유.** 상수를 더하거나 빼는 것은 크리틱이 그대로
   흡수해 버려서 이점이 바뀌지 않는다. 판 단위 베이스라인이 실제로 효과를 내려면
   에피소드 전체 결과가 하나의 신호로 들어와야 한다. 대신 에피소드 안에서
   '어느 수가 좋았나'는 크리틱 V(s_t) 가 맡는다 (train.py 는 γ=1, λ=1 이라
   A_t = R - V(s_t) 인 순수 몬테카를로 이점이 된다 — 부트스트랩 사슬 0홉).

**평가는 영향받지 않는다.** measure.py 는 make_env() 를 쓰고 그룹/보상 로직은
make_train_env() 쪽에만 있다. 점수는 언제나 게임 원래 규칙대로 잰다.

주의: 이 환경에서 `rollout/ep_rew_mean` 은 0 근처를 맴돈다(정의상 그렇다).
학습이 되는지는 **`rollout/ep_score_mean`** 으로 봐야 한다.
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

# 한 판을 몇 번 반복할 것인가. K가 크면 베이스라인이 정확해지고, 작으면 같은
# 시간에 더 다양한 판을 본다. 8 이면 6M 스텝에 약 15,000개의 서로 다른 판을 본다.
GROUP_SIZE = 8

# 상대 점수를 나누는 값. 실측한 '판 안 std'(5.41)에 맞춰 이점이 O(1) 이 되게 한다.
# PPO 의 normalize_advantage 가 미니배치 단위로 다시 정규화하므로 정확할 필요는 없고,
# 가치 손실의 크기를 정책 손실과 비슷한 자릿수로 맞추는 역할이 크다.
GROUP_SCORE_SCALE = 6.0

# 첫 롤아웃용 전역 평균의 EMA 계수. 판마다 새로 뽑히므로 너무 빠르면 잡음을 탄다.
GLOBAL_MEAN_MOMENTUM = 0.99

# ── 탐색 설정 (runs/agents/ai/search.py 가 읽는다) ────────────────────────
SEARCH_TOP_K = 0    # 순수 신경망. 추론은 1회 forward.
SEARCH_DEPTH = 2
SEARCH_BEAM_WIDTH = 6
SEARCH_BEAM_TOP_K = 16


class AppleGameEnv(AppleGameEnvBase):
    """평가용 기본 환경. 보상은 점수/162 그대로 (셰이핑 없음)."""

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

        digits = np.arange(N_DIGITS, dtype=np.int8)[:, None, None]
        obs[:N_DIGITS] = (self.board.grid[None, :, :] == digits)

        density = self._option_density()
        peak = float(density.max())
        obs[N_DIGITS] = density / peak if peak > 0 else 0.0
        obs[N_DIGITS + 1] = np.minimum(np.log1p(density) / DENSITY_LOG_SCALE, 1.0)
        obs[N_DIGITS + 2] = min(len(self.board.get_valid_actions()) / VALID_ACTION_SCALE, 1.0)

        return obs

    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        return (remove_count / self.total_cell_count) * self.STEP_REWARD_TOTAL

    def invalid_action_reward(self) -> float:
        return self.INVALID_ACTION_REWARD


class GroupRelativeEnv(AppleGameEnv):
    """같은 판을 repeat 번 두고, 보상을 **그 판 안에서의 상대 점수**로 준다. 학습 전용.

    board_source 를 명시적으로 주면(평가·디버깅) 그 판을 그대로 쓰고 그룹 로직은
    비켜간다. 그래서 seed 를 지정한 재현 실행은 그대로 된다.

    info 로 내보내는 것 (train.py 의 진단 콜백이 읽는다):
        group_index      그룹 번호 (환경마다 독립적으로 증가)
        group_slot       그룹 안에서 몇 번째 판인가 (1..repeat)
        group_baseline   이 에피소드에 쓰인 베이스라인
    """

    def __init__(self, *args, repeat: int = GROUP_SIZE,
                 score_scale: float = GROUP_SCORE_SCALE, **kwargs):
        super().__init__(*args, **kwargs)
        self._repeat = repeat
        self._score_scale = score_scale
        self._left = 0
        self._board_seed: int | None = None
        self._group_index = -1
        self._group_slot = 0
        self._group_scores: list[float] = []
        self._global_mean: float | None = None
        self._baseline_used = 0.0
        # 환경마다(=프로세스마다) 서로 다른 판을 보도록 OS 엔트로피로 초기화한다.
        self._board_rng = np.random.default_rng()

    # ── 판 공급 ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if options.get("board_source") is None:
            if self._left <= 0:
                self._board_seed = int(self._board_rng.integers(0, 2 ** 31 - 1))
                self._left = self._repeat
                self._group_index += 1
                self._group_slot = 0
                self._group_scores = []
            options["board_source"] = self._board_seed
            self._left -= 1
            self._group_slot += 1
        return super().reset(seed=seed, options=options)

    # ── 보상 ─────────────────────────────────────────────────────────────
    def _baseline(self) -> float:
        """현재 에피소드의 결과에 의존하지 않는 값이어야 편향이 없다."""
        if self._group_scores:
            return float(np.mean(self._group_scores))       # 이 판의 이전 롤아웃들
        if self._global_mean is not None:
            return self._global_mean                        # 그룹의 첫 판 -> 전역 평균
        return float(self.score)                            # 이 환경의 맨 첫 판 -> 보상 0

    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        if not terminated:
            return 0.0

        score = float(self.score)
        baseline = self._baseline()
        self._baseline_used = baseline

        self._group_scores.append(score)
        self._global_mean = (score if self._global_mean is None
                             else GLOBAL_MEAN_MOMENTUM * self._global_mean
                             + (1.0 - GLOBAL_MEAN_MOMENTUM) * score)

        return (score - baseline) / self._score_scale

    def _get_info(self):
        info = super()._get_info()
        info["group_index"] = self._group_index
        info["group_slot"] = self._group_slot
        info["group_baseline"] = self._baseline_used
        return info


def make_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """평가·측정용. 매 판 새로운 무작위 판, 보상은 점수/162."""
    return AppleGameEnv(rows=rows, cols=cols, render_mode=render_mode)


def make_train_env(rows: int, cols: int, render_mode: str | None = None) -> AppleGameEnv:
    """학습용. 같은 판을 GROUP_SIZE 번 반복하고 그룹 상대 보상을 준다."""
    return GroupRelativeEnv(rows=rows, cols=cols, render_mode=render_mode,
                            repeat=GROUP_SIZE, score_scale=GROUP_SCORE_SCALE)
