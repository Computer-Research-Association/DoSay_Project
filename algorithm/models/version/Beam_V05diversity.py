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

@registry.heuristic(weight=1.0)
def feature_action_count(board: Board) -> float:
    return sigmoid(len(board.get_valid_actions()), 0.2, 15.0)

@registry.heuristic(weight=1.0)
def feature_action_diversity(board: Board) -> float:
    acts = board.get_valid_actions()
    if not acts: return 0.0
    g = board.grid
    digits = set()
    for a in acts:
        (r1,c1),(r2,c2) = a.top_left, a.bottom_right
        region = g[r1:r2+1, c1:c2+1]
        digits.update(int(v) for v in np.unique(region) if v > 0)
    return len(digits) / 9.0
