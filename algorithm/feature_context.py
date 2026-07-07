import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from game.board import compute_prefix_sum
from game.action import Action
from game.board import Board


@dataclass 
class FeatureContext:
    board: NDArray[np.int8]
    prefix: NDArray
    count_by_value: dict[int, int]
    valid_actions: list[Action]

    @classmethod 
    def from_board(cls, board_obj: Board) -> "FeatureContext":
        grid = board_obj.grid  # Board 객체 안의 실제 배열
        prefix = compute_prefix_sum(grid) #누적합
        counts = {v: int((grid == v).sum()) for v in range(1, 10)}
        valid_actions = board_obj.get_valid_actions()  
        return cls(board=grid, prefix=prefix, count_by_value=counts, valid_actions=valid_actions)
    
    

    

    