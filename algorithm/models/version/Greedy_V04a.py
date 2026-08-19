from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np

registry = HeuristicRegistry()


@registry.heuristic(weight=1.0)
def feature_remove_nine(board: Board) -> float:
    grid = board.grid
    n9 = int((grid == 9).sum())
    if n9 == 0:
        return 0.0
    n1 = int((grid == 1).sum())
    if n1 == 0:
        return -1.0                       # 9는 있는데 짝지을 1이 없음 = 최악
    return -sigmoid(n9 / n1, 2.5, 1.0)    # 9가 1보다 많을수록 페널티
