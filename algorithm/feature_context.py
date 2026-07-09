import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass, field
from game.board import compute_prefix_sum
from game.action import Action
from game.board import Board



@dataclass 
class FeatureContext:
    board_array: NDArray[np.int8] # 숫자 배열의 좌푯값
    board_obj : Board #Board 인스턴트
    prefix: NDArray  # 누적합
    count_by_value: dict[int, int]  #숫자별 남아있는 개수
    action : Action
    _valid_actions: list | None = field(default=None, repr=False)
    
    @property
    def valid_actions(self): #valid action 함수 불러오기
        if self._valid_actions is None:
            self._valid_actions = self.board_obj.get_valid_actions(self.board)  # next_board 기준으로 딱 1번
        return self._valid_actions

    @classmethod 
    def from_board(cls, board_obj:Board, action) -> "FeatureContext":
        board_array = board_obj.board
        prefix = compute_prefix_sum(board_array)
        counts = {v: int((board_array == v).sum()) for v in range(1, 10)}
        return cls(board=board_array, prefix = prefix, count_by_value = counts, action = action )