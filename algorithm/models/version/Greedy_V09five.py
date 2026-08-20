from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

registry = HeuristicRegistry()

def _ratio_penalty(big, small):
    if big == 0: return 0.0
    if small == 0: return -1.0
    return -sigmoid(big / small, 2.5, 1.0)

def _center_closeness(grid):
    H, W = grid.shape
    cr, cc = (H-1)/2, (W-1)/2
    ys, xs = np.nonzero(grid)
    if len(ys) == 0: return None
    dist = np.sqrt((ys-cr)**2 + (xs-cc)**2)
    return float(np.mean(1 - dist/np.sqrt(cr**2+cc**2)))

@registry.heuristic(weight=1.0)
def feature_remove_nine(board: Board) -> float:
    g = board.grid
    return _ratio_penalty(int((g==9).sum()), int((g==1).sum()))

@registry.heuristic(weight=1.0)
def feature_remove_eight(board: Board) -> float:
    g = board.grid
    return _ratio_penalty(int((g==8).sum()), int((g==2).sum()))

@registry.heuristic(weight=1.0)
def feature_action_count(board: Board) -> float:
    return sigmoid(len(board.get_valid_actions()), 0.2, 15.0)

@registry.heuristic(weight=1.0)
def feature_remove_five(board: Board) -> float:
    n5 = int((board.grid==5).sum())
    return 0.0 if n5==0 else -sigmoid(n5, 0.5, 4.0)
