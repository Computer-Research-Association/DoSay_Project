from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np

registry = HeuristicRegistry()

@registry.heuristic(weight=3.0)
def feature_remove_nine(board: Board) -> float:
    area = board.grid
    if not(area == 9).any():
        return 0.0
    nine_count = int((area==9).sum())
    if not _has_nine_one_pair(board):
        return 0.0
    return -nine_count / 10

@registry.heuristic(weight=1.0)
def feature_remove_eight(board: Board) -> float:
    area = board.grid
    if not(area == 8).any():
        return 0.0
    eight_count = int((area==8).sum()) 
    if not _has_eight_pair(board):
        return 0.0
    return -eight_count / 10

@registry.heuristic(weight=1.0)
def feature_spread(board: Board) -> float:
    remaining = (board.grid != 0).astype(np.int8)   
    n = int(remaining.sum())
    if n == 0:
        return 0.0                         
    neighbors = np.zeros_like(remaining)
    neighbors[1:, :]  += remaining[:-1, :]            # 위
    neighbors[:-1, :] += remaining[1:, :]             # 아래
    neighbors[:, 1:]  += remaining[:, :-1]            # 왼
    neighbors[:, :-1] += remaining[:, 1:]             # 오

    clumping = int((neighbors * remaining).sum())   
    return 1.0 - clumping / (4 * n)             

'''@registry.heuristic(weight=1.0)
def feature_remove_seven(board: Board) -> float:
    area = board.grid
    if not(area == 7).any(): 
        return 0.0
    seven_count = int((area==7).sum()) 
    if not _has_seven_pair(board): 
        return 0.0
    return sigmoid(seven_count, k = 0.5, x0 = 3.0)
    '''

'''
@registry.heuristic(weight=1.0)
def feature_remove_six(board: Board) -> float:
    area = board.grid
    if not(area == 6).any():
        return 0.0
    six_count = int((area==6).sum())
    if not _has_six_pair(board):
        return 0.0
    return sigmoid(six_count, k = 0.5, x0 = 3.0) 
'''

'''
@registry.heuristic(weight=1.0)
def feature_remove_five(board: Board) -> float:
    area = board.grid
    if not(area == 5).any():
        return 0.0
    five_count = int((area==5).sum()) 
    if not _has_five_pair(board):
        return 0.0
    return sigmoid(five_count, k = 0.5, x0 = 3.0)
'''


def _has_nine_one_pair(board: Board) -> bool:
    for action in board.get_valid_actions():
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = board.grid[r1:r2+1, c1:c2+1]
        if (region == 9).any():
            return True
    return False

def _has_eight_pair(board: Board) -> bool:
    for action in board.get_valid_actions():
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = board.grid[r1:r2+1, c1:c2+1]
        if (region == 8).any():
            return True
    return False

def _has_seven_pair(board: Board) -> bool:
    for action in board.get_valid_actions():
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = board.grid[r1:r2+1, c1:c2+1]
        if (region == 7).any():
            return True
    return False

def _has_six_pair(board: Board) -> bool:

    for action in board.get_valid_actions():
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = board.grid[r1:r2+1, c1:c2+1]
        if (region == 6).any():
            return True
    return False

def _has_five_pair(board: Board) -> bool:
    for action in board.get_valid_actions():
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = board.grid[r1:r2+1, c1:c2+1]
        if (region == 5).any():
            return True
    return False

'''
@registry.heuristic(weight=1.0)   
def feature_remove_the_most_grouping(board: Board) -> float:
    return float(board.grid != 0).sum() / board.grid.size
'''
