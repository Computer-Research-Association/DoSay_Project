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
    def __init__(self, init_board: NDArray[np.int8] | None = None, board_size: tuple[int, int] | None = None, seed = None):
        self.sum_prefix: NDArray[np.int32]
        self.grid: NDArray[np.int8]
        self.size: tuple[int, int]

        if seed is not None:
            np.random.seed(seed)

        if init_board is None and board_size is None:
            raise Exception("Both board_size and init_board cannot be None.")
        elif init_board is None:
            self.size = board_size # type: ignore
            self.grid = _generate_board(self.size)
        elif board_size is None:
            self.size = init_board.shape
            self.grid = init_board
        else:
            if init_board.shape != board_size:
                raise Exception("board_size does not match init_board.size.")
            self.size = board_size
            self.grid = init_board

        R, C = self.size
        self.sum_prefix = np.empty((R+1, C+1), dtype=np.int32)
        self._update_prefix()


    def _update_prefix(self):
        self.sum_prefix = compute_prefix_sum(self.grid)

    def do_action(self, action: Action) -> None:
        # if not self.is_valid_action(action):
        #     return False
        (r1, c1), (r2, c2) = action.top_left, action.bottom_right
        self.grid[r1:r2+1, c1:c2+1] = 0
        self._update_prefix()

    def _get_area_sum(self, top_left: tuple[int, int], bottom_right: tuple[int, int]) -> int:
        (r1, c1), (r2, c2) = top_left, bottom_right
        return (self.sum_prefix[r2 + 1, c2 + 1]
                - self.sum_prefix[r1, c2 + 1]
                - self.sum_prefix[r2 + 1, c1]
                + self.sum_prefix[r1, c1])

    # 아 몰라 일단 구현해!!!!!!!!!
    def get_valid_actions(self) -> list[Action]:
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

        return valid_actions
    
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
    
    # def is_valid_action(self, action: Action) -> bool:
    #     area = self.get_area(action)
    #     action_valid = area.sum() == 10

    #     return action_valid
    
    def print_board(self):
        for line in self.grid:
            print(*(f"{v:1d}" for v in line))

    def is_done(self) -> tuple[bool, bool]:
        actions = self.get_valid_actions()

        is_over = len(actions) == 0
        is_all_clear = self.grid.sum() == 0

        return (is_over, is_all_clear)
    
def _generate_board(size) -> NDArray[np.int8]:
    _board = np.random.randint(1, 10, size=size, dtype=np.int8)
    return _board

def slice_area(board_array: NDArray, action: "Action") -> NDArray:
    r1, c1 = action.top_left
    r2, c2 = action.bottom_right
    return board_array[r1:r2+1, c1:c2+1]