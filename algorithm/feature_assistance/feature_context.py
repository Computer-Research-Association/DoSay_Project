import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass, field
from game.board import compute_prefix_sum
from game.action import Action
from game.board import Board



@dataclass 
class FeatureContext:
    board_array: NDArray[np.int8] # 숫자 배열의 좌푯값
    prefix: NDArray  # 누적합
    count_by_value: dict[int, int]  #숫자별 남아있는 개수
    action : Action
    

    _area: NDArray | None = field(default= None, repr = False)
    _board_after_action : NDArray | None = field(default= None, repr = False)
    _valid_actions: list | None = field(default=None, repr=False)

    @property
    def valid_actions(self): #valid action 함수 불러오기
        if self._valid_actions is None:
            self._valid_actions = self.board_obj.get_valid_actions()  # next_board 기준으로 딱 1번
        return self._valid_actions
    
    @property
    def board_after_action(self): 
        if self._board_after_action is None:
            r1, c1 = self.action.top_left
            r2, c2 = self.action.bottom_right
            temp = self.board_array.copy()
            temp[r1:r2+1, c1:c2+1] = 0
            self._board_after_action = temp
 
        return self._board_after_action
    
    @property
    def area(self):
        if self._area is None:
            r1, c1 = self.action.top_left
            r2, c2 = self.action.bottom_right
            self._area = self.board_array[r1:r2+1, c1:c2+1]
        return self._area


    @classmethod 
    def from_board(cls, board_array, action, valid_actions) -> "FeatureContext":
        prefix = compute_prefix_sum(board_array)
        counts = {v: int((board_array == v).sum()) for v in range(1, 10)}
        return cls(board_array=board_array, prefix = prefix, count_by_value = counts, action = action, _valid_actions = valid_actions)