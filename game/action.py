from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]