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
def feature_isolated_nine(board: Board) -> float:
    g = board.grid
    n9 = int((g==9).sum())
    if n9 == 0: return 0.0
    ones = (g==1).astype(np.int8)
    # 각 셀 주변 3x3에 1이 있는지 (경계 패딩)
    pad = np.pad(ones, 1)
    has_one_near = sliding_window_view(pad, (3,3)).sum(axis=(2,3)) > 0
    isolated = int(((g==9) & (~has_one_near)).sum())   # 근처에 1 없는 9
    return -isolated / n9        # 고립 비율만큼 페널티
