from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np

registry = HeuristicRegistry()

def _ratio_penalty(big, small):
    if big == 0: return 0.0
    if small == 0: return -1.0
    return -sigmoid(big / small, 2.5, 1.0)

# ★ 9 무조건 우선: 압도적 가중치로 '남은 9 개수'를 최우선 최소화
@registry.heuristic(weight=100.0)
def feature_nine_priority(board: Board) -> float:
    return -int((board.grid == 9).sum())

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
