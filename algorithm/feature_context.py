import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from game.board import compute_prefix_sum
from game.action import Action


@dataclass 
class FeatureContext:
    board: NDArray[np.int8] # 숫자 배열의 좌푯값
    prefix: NDArray  # 누적합
    count_by_value: dict[int, int]  #숫자별 남아있는 개수
    action: Action # 현재 평가하고 있는 액션

    @classmethod 
    def from_board(cls, board: NDArray[np.int8], action: Action) -> "FeatureContext":
        prefix = compute_prefix_sum(board)
        counts = {v: int((board == v).sum()) for v in range(1, 10)}
        return cls(board=board, prefix = prefix, count_by_value = counts, action = action)
    
    

    

    