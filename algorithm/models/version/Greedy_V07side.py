from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np

registry = HeuristicRegistry()

def _ratio_penalty(big, small):
    if big == 0: return 0.0
    if small == 0: return -1.0
    return -sigmoid(big / small, 2.5, 1.0)

@registry.heuristic(weight=1.0)
def feature_remove_nine(board: Board) -> float:
    g = board.grid
    return _ratio_penalty(int((g==9).sum()), int((g==1).sum()))

@registry.heuristic(weight=1.0)
def feature_remove_eight(board: Board) -> float:
    g = board.grid
    return _ratio_penalty(int((g==8).sum()), int((g==2).sum()))

def _center_closeness(grid):
    """남은 셀의 '중앙 가까움' 평균 (0~1). 중앙일수록 1, 모서리일수록 0."""
    H, W = grid.shape
    cr, cc = (H - 1) / 2, (W - 1) / 2
    ys, xs = np.nonzero(grid)
    if len(ys) == 0:
        return None
    dist = np.sqrt((ys - cr)**2 + (xs - cc)**2)
    max_dist = np.sqrt(cr**2 + cc**2)
    return float(np.mean(1 - dist / max_dist))

@registry.heuristic(weight=1.0)
def feature_side(board: Board) -> float:
    c = _center_closeness(board.grid)
    return 0.0 if c is None else (1.0 - c)    # 모서리 가까울수록 ↑
