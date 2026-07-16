from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from .action import Action

def compute_prefix_sum(grid: NDArray) -> NDArray:
    height, width = grid.shape
    prefix = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.cumsum(grid, axis=0, out=prefix[1:, 1:])
    np.cumsum(prefix[1:, 1:], axis=1, out=prefix[1:, 1:])
    return prefix

class Board():
    def __init__(self, _grid: NDArray[np.int8]):
        """직접 호출하지 말고 from_seed/from_seed 사용"""
        self.grid: NDArray[np.int8] = _grid.astype(np.int8).copy()
        self.size: Tuple[int, int] = _grid.shape

        sum_prefix_size = (self.size[0]+1, self.size[1]+1)
        self._sum_prefix: NDArray[np.int32] = np.empty(sum_prefix_size, dtype=np.int32)
        self._update_sum_prefix()
        
        self._valid_actions_cache: list[Action]
        self._update_valid_actions()

    @classmethod
    def from_board(cls, _board: NDArray[np.int8]):
        return cls(_board)

    @classmethod
    def from_seed(cls, size: Tuple[int, int], seed: int | None = None):
        rng = np.random.default_rng(seed)
        _board = rng.integers(1, 10, size=size, dtype=np.int8)
        return cls(_board)

    def get_sum_prefix(self) -> NDArray[np.int32]:
        return self._sum_prefix

    def _update_sum_prefix(self) -> None:
        self._sum_prefix = compute_prefix_sum(self.grid)

    def do_action(self, action: Action) -> bool:
        if not self.is_valid_action(action):
            return False
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        self.grid[r1:r2+1, c1:c2+1] = 0
        self._update_sum_prefix()
        self._update_valid_actions()  # 추가: 그리드 변경 시에만 재계산
        return True

    def _get_area_sum(self, top_left: tuple[int, int], bottom_right: tuple[int, int]) -> int:
        (r1, c1), (r2, c2) = top_left, bottom_right
        return (self._sum_prefix[r2 + 1, c2 + 1]
                - self._sum_prefix[r1, c2 + 1]
                - self._sum_prefix[r2 + 1, c1]
                + self._sum_prefix[r1, c1])

    def get_valid_actions(self) -> list[Action]:
        return self._valid_actions_cache

    def _update_valid_actions(self) -> None:
        valid_actions = []
        for r1 in range(self.size[0]):
            for r2 in range(r1, self.size[0]):
                for c1 in range(self.size[1]):
                    for c2 in range(c1, self.size[1]):
                        area_sum = self._get_area_sum((r1, c1), (r2, c2))

                        if area_sum == 10:
                            if self.is_smallest_action((r1, c1), (r2, c2)):
                                valid_actions.append(Action((r1, c1), (r2, c2)))
                            else:
                                continue
                        elif area_sum > 10:
                            break

        self._valid_actions_cache = valid_actions

    
    def is_smallest_action(self, top_left: tuple[int, int], bottom_right: tuple[int, int]) -> bool:
        (r1, c1), (r2, c2) = top_left, bottom_right

        top_sum = self._get_area_sum((r1, c1), (r1, c2))
        if top_sum == 0: return False

        right_sum = self._get_area_sum((r1, c2), (r2, c2))
        if right_sum == 0: return False

        bottom_sum = self._get_area_sum((r2, c1), (r2, c2))
        if bottom_sum == 0: return False

        left_sum = self._get_area_sum((r1, c1), (r2, c1))
        if left_sum == 0: return False

        return True
    
    def is_valid_action(self, action: Action) -> bool:
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right

        area = self.grid[r1:r2+1, c1:c2+1]
        action_valid = area.sum() == 10

        return action_valid
    
    def print_board(self, hideZero: bool = True):
        if hideZero:
            for line in self.grid:
                print(*(" " if v == 0 else f"{v:1d}" for v in line))
        else:
            for line in self.grid:
                print(*(f"{v:1d}" for v in line))

    def is_done(self) -> tuple[bool, bool]:
        actions = self.get_valid_actions()

        is_over = len(actions) == 0
        is_all_clear = self.grid.sum() == 0

        return (is_over, is_all_clear)
    
    def slice_area(self, action: Action) -> NDArray[np.int8]:
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        return self.grid[r1:r2+1, c1:c2+1]