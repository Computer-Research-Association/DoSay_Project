from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np

registry = HeuristicRegistry()

def _ratio_penalty(big, small):        # big가 small보다 많을수록 페널티
    if big == 0:
        return 0.0
    if small == 0:
        return -1.0
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
def feature_remove_seven(board: Board) -> float:
    g = board.grid
    return _ratio_penalty(int((g==7).sum()), int((g==3).sum()))

@registry.heuristic(weight=1.0)
def feature_remove_five(board: Board) -> float:
    n5 = int((board.grid==5).sum())     # 5+5 자기짝 → 남은 5 적을수록 좋음
    if n5 == 0:
        return 0.0
    return -sigmoid(n5, 0.5, 4.0)
