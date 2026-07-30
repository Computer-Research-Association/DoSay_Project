"""사과게임 Gymnasium 환경의 **공유 골격**.

이 파일에는 학습 결과에 영향을 주는 코드가 한 줄도 없다. 담당하는 것은
**하네스 계약**뿐이다:

  - reset(options={"board_source": ...}) 시그니처
  - info 딕셔너리의 키 (action_mask / step_num / score)
  - step() 의 5-tuple 반환 규약과 진행 순서
  - render_mode 처리
  - 액션 마스킹 전제(Discrete + index_to_action 매핑)

이 부분은 runs/measure.py 같은 벤치마크 하네스가 의존하는 인터페이스이며,
바꾸더라도 **이미 학습된 가중치에는 영향이 없다**. 오히려 모든 버전이 함께
따라와야 벤치마크가 계속 돌아가므로 공유하는 편이 맞다.

반대로 **학습 결과를 결정하는 것**은 전부 추상 메서드로만 존재한다.
기본 구현을 두지 않는 이유: 기본값이 있으면 결국 그걸 고치게 되고, 그 순간
과거 버전의 학습 조건이 조용히 바뀐다. 구현하지 않으면 인스턴스를 만들 수조차
없으므로 실수로 상속만 하고 넘어가는 일이 생기지 않는다.

버전이 반드시 구현해야 하는 것:
    build_actions              행동 집합       (ai/envs/action_sets.py 에서 골라 쓴다)
    build_observation_space    관측 공간
    _get_obs                   관측 인코딩
    compute_reward             보상 공식
    invalid_action_reward      무효수 페널티

사용법: ai/models/version/<버전>/env.py 에서 이 클래스를 상속해 위 5개를 구현하고,
make_env(rows, cols, render_mode) 팩토리를 노출한다.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from game.GUI import GUI
from game.action import Action
from game.board import Board


class AppleGameEnvBase(gym.Env, ABC):
    metadata = {
        "render_modes": ["human", "ansi", "None"]
    }

    def __init__(self, rows: int, cols: int, render_mode: str | None = None):
        super().__init__()
        self.row_num = rows
        self.col_num = cols
        self.total_cell_count = rows * cols

        if render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode
            if render_mode == "human":
                self.GUI = GUI(rows, cols, title="AppleGame - AI")
        else:
            self.render_mode = "ansi"

        self.board: Board
        self.last_action: Action | None = None
        self.step_num = 0
        self.score = 0

        # action idx로 처리하기 위해 초기 매핑 진행. Descrete Action용
        self.index_to_action: list[Action] = self.build_actions(rows, cols)
        self.action_to_index: dict[Action, int] = { act: idx for idx, act in enumerate(self.index_to_action) }

        # 입/출력층(input/output layer)
        self.observation_space = self.build_observation_space(rows, cols)
        self.action_space = spaces.Discrete(len(self.index_to_action))

    # ── 버전이 반드시 구현하는 것 (학습 결과를 결정하는 부분) ─────────────────

    @abstractmethod
    def build_actions(self, rows: int, cols: int) -> list[Action]:
        """행동 집합. 바뀌면 action_space 크기와 로짓 순서가 달라진다."""

    @abstractmethod
    def build_observation_space(self, rows: int, cols: int) -> spaces.Space:
        """관측 공간. _get_obs 의 반환값과 반드시 일치해야 한다."""

    @abstractmethod
    def _get_obs(self) -> NDArray:
        """현재 보드를 관측으로 인코딩한다. 바뀌면 인코더 입력이 달라진다."""

    @abstractmethod
    def compute_reward(self, *, remove_count: int, terminated: bool, is_all_clear: bool) -> float:
        """한 수에 대한 보상. 필요한 상수는 각 버전이 자기 클래스 안에 정의한다."""

    @abstractmethod
    def invalid_action_reward(self) -> float:
        """둘 수 없는 수를 골랐을 때의 보상. 액션 마스킹이 걸려 있으면 도달하지 않는다."""

    # ── 하네스 계약 (공유) ─────────────────────────────────────────────────

    def get_action_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(len(self.index_to_action), dtype=np.bool_)
        for action in self.board.get_valid_actions():
            mask[self.action_to_index[action]] = True
        return mask

    def _get_info(self) -> dict[str, Any]:
        return {
            "action_mask": self.get_action_mask(),
            "step_num": self.step_num,
            "score": self.score
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        shape = (self.row_num, self.col_num)

        board_source = options.get("board_source")
        self.last_action = None
        self.step_num = 0
        if isinstance(board_source, (np.ndarray, list)):
            _board = np.asarray(board_source, dtype=np.int8)
            self.board = Board.from_board(_board)
            self.score = _board.size - int(np.count_nonzero(_board))
        elif board_source is None or isinstance(board_source, int):
            self.board = Board.from_seed(shape, board_source)
            self.score = 0
        else:
            raise TypeError(f"board_source must be ndarray, list, or int, got {type(board_source).__name__}")

        return self._get_obs(), self._get_info()

    def step(self, action: int):  # action_idx -> action
        act = self.index_to_action[action]
        is_valid, remove_count = self.board.do_action(act)
        if not is_valid:
            return self._get_obs(), self.invalid_action_reward(), False, False, self._get_info()

        self.last_action = act
        self.step_num += 1
        self.score += remove_count

        terminated, is_all_clear = self.board.is_done()
        reward = self.compute_reward(
            remove_count=remove_count, terminated=terminated, is_all_clear=is_all_clear
        )

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode in self.metadata["render_modes"]:
            if self.render_mode == "human":
                self.GUI.render(self.board.grid, score=self.score, step=self.step_num,
                                remaining=int(np.count_nonzero(self.board.grid)), action=self.last_action)
            elif self.render_mode == "ansi":
                self.board.print_board(True)
            elif self.render_mode == "None":
                return None
        return None
