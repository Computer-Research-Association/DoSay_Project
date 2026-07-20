from typing import Any

import gymnasium as gym
from gymnasium import spaces
from game.action import Action
from game.board import Board
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from . import constants

def get_all_action(rows, cols) -> list[Action]:
    actions = []
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    if r1 == r2 and c1 == c2: continue  #  1x1 사이즈인 경우 제외
                    actions.append(Action((r1, c1), (r2, c2)))
    return actions


class AppleGameEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi", "None"], # "human" 은 추후 개발
    }


    def __init__(self, rows: int, cols: int, render_mode: str | None = None):
        self.row_num = rows
        self.col_num = cols
        self.total_cell_count = rows * cols
        self.render_mode = render_mode
        super().__init__()

        self.board: Board
        self._score = 0

        # action idx로 처리하기 위해 초기 매핑 진행. Descrete Action용
        self.index_to_action: list[Action] = get_all_action(rows, cols)
        self.action_to_index: dict[Action, int] = { act: idx for idx, act in enumerate(self.index_to_action) }
        
        # 입/출력층(input/output layer)
        self.observation_space = spaces.Box(low=0, high=9, shape=(rows, cols), dtype=np.int8)
        self.action_space = spaces.Discrete(len(self.index_to_action))


    def _get_obs(self) -> NDArray[np.int8]:
        return self.board.grid.copy()
    
    def get_action_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(len(self.index_to_action), dtype=np.bool_)
        # self.action_to_index[Action((0, 2), (0, 3))] = 12
        for action in self.board.get_valid_actions():
            # idx = self.action_to_index[Action((0, 2), (0, 3))]
            idx = self.action_to_index[action]
            # print("g")
            mask[idx] = True
        return mask

    def _get_info(self) -> dict[str, Any]:
        return {
            "action_mask": self.get_action_mask(),
            "score": self._score
        }
    

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        shape = (self.row_num, self.col_num)

        board_source = options.get("board_source")

        if isinstance(board_source, (np.ndarray, list)):
            _board = np.asarray(board_source, dtype=np.int8)
            self.board = Board.from_board(_board)
            self._score = _board.size - np.count_nonzero(_board)
        elif board_source is None or isinstance(board_source, int):
            self.board = Board.from_seed(shape, board_source)
            self._score = 0
        else:
            raise TypeError(f"board_source must be ndarray, list, or int, got {type(board_source).__name__}")

        obs = self._get_obs()
        info = self._get_info()
        
        # if self.render_mode == "human":
        #     self._render_frame()

        return obs, info


    def step(self, action: int):  # action_idx -> action
        act = self.index_to_action[action]        
        is_valid, remove_count = self.board.do_action(act)
        if not is_valid:
            # print("경고: 로직상 도달하면 안되는 영역의 코드가 작동됨. [ ai > env > step() > valid ]")
            obs = self._get_obs()
            info = self._get_info()
            return obs, -1.0, False, False, info

        reward = (remove_count/self.total_cell_count) * constants.STEP_REWARD_TOTAL
        self._score += remove_count

        terminated, is_all_clear = self.board.is_done()

        if terminated:
            if is_all_clear:
                reward += constants.REWARD_ALL_CLEAR_BONUS
            reward += (self._score/self.total_cell_count) * constants.TERMINAL_REWARD_TOTAL

        obs = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode in self.metadata["render_modes"]:
            if self.render_mode == "ansi":
                self.board.print_board(True)
            if self.render_mode == "None":
                return None
        return None