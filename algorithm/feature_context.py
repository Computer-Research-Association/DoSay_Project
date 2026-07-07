import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from game.board import compute_prefix_sum
from game.action import Action
from game.board import Board

@dataclass 
class FeatureContext:
    board: NDArray[np.int8] # 숫자 배열의 좌푯값
    prefix: NDArray  # 누적합
    count_by_value: dict[int, int]  #숫자별 남아있는 개수
    action : Action
    _valid_actions: list | None = field(default=None, repr=False)
    
    @property
    def valid_actions(self): #valid action 함수 불러오기
        if self._valid_actions is None:
            self._valid_actions = Board.get_valid_actions(self.board)  # next_board 기준으로 딱 1번
        return self._valid_actions

    @classmethod 
    def from_board(cls, board: NDArray[np.int8]) -> "FeatureContext":
        prefix = compute_prefix_sum(board)
        counts = {v: int((board == v).sum()) for v in range(1, 10)}
        return cls(board=board, prefix = prefix, count_by_value = counts)
    
    

    

    