from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    # (row, col)
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]

def slice_area(board_array: NDArray, action: "Action") -> NDArray:
    r1, c1 = action.top_left
    r2, c2 = action.bottom_right
    return board_array[r1:r2+1, c1:c2+1]