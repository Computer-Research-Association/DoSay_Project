from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry
from game.board import Board
import numpy as np

registry = HeuristicRegistry()


@registry.heuristic(weight=3.0)
def feature_remove_nine(board: Board) -> float:
    grid = board.grid
    nine_count = int((grid == 9).sum())
    if nine_count == 0:
        return 0.0                 

    one_count = int((grid == 1).sum())
    if one_count == 0:
        return -1.0                     

    ratio = nine_count / one_count 
    print(f"ratio = {ratio: .2f}")    
    return -sigmoid(ratio, 2.0, 0.8)


@registry.heuristic(weight=2.0)
def feature_remove_eight(board: Board) -> float:
    grid = board.grid
    eight_count = int((grid == 8).sum())
    if eight_count == 0:
        return 0.0

    one_count = int((grid == 1).sum())
    two_count = int((grid == 2).sum())

    ratio = eight_count / (one_count + two_count)

    

'''
@registry.heuristic(weight=1.0)
def feature_remove_eight(board: Board) -> float:
    area = board.grid
    if not(area == 8).any():
        return 0.0
    eight_count = int((area==8).sum()) 
    if not _has_eight_pair(board):
        return 0.0
    return -eight_count / 10
'''

'''
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
'''

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


'''
@registry.heuristic(weight=1.0)   
def feature_remove_the_most_grouping(board: Board) -> float:
    return float(board.grid != 0).sum() / board.grid.size
'''
