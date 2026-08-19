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
        return -1.0
    return -sigmoid(n9 / n1, 2.5, 1.0)


@registry.heuristic(weight=1.0)
def feature_remove_eight(board: Board) -> float:
    grid = board.grid
    n8 = int((grid == 8).sum())
    if n8 == 0:
        return 0.0
    n1 = int((grid == 1).sum())
    n2 = int((grid == 2).sum())
    n9 = int((grid == 9).sum())
    supply = n2 + n1
    if supply == 0:
        return -1.0
    return -sigmoid(n8 / supply, 2.5, 1.0)
