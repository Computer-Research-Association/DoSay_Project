from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    # (row, col)
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]