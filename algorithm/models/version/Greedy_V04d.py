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


@registry.heuristic(weight=1.0)
def feature_remove_eight(board: Board) -> float:
    grid = board.grid
    n8 = int((grid == 8).sum())
    if n8 == 0:
        return 0.0
    n2 = int((grid == 2).sum())
    if n2 == 0:
        return -1.0                       # 8은 있는데 짝지을 2가 없음
    return -sigmoid(n8 / n2, 2.5, 1.0)


@registry.heuristic(weight=1.0)
def feature_spread(board: Board) -> float:
    occ = (board.grid != 0).astype(np.int8)
    n = int(occ.sum())
    if n == 0:
        return 0.0
    nb = np.zeros_like(occ)
    nb[1:, :]  += occ[:-1, :]
    nb[:-1, :] += occ[1:, :]
    nb[:, 1:]  += occ[:, :-1]
    nb[:, :-1] += occ[:, 1:]
    clumping = int((nb * occ).sum())
    return 1.0 - clumping / (4 * n)       # 퍼질수록 높음(좋음)
