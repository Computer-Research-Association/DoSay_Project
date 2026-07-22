from algorithm.feature_assistance.feature_context import FeatureContext
from algorithm.feature_assistance.math_algorithm import sigmoid
from algorithm.models.utils import HeuristicRegistry

registry = HeuristicRegistry()

@registry.heuristic(weight=1.0)
def feature_remove_nine(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 9).any():
        return 0.0
    nine_count = int((area==9).sum())
    if not _has_nine_one_pair(ctx):
        return 0.0
    return sigmoid(nine_count, k = 0.5, x0 = 3.0)

@registry.heuristic(weight=1.0)
def feature_remove_eight(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 8).any():
        return 0.0
    eight_count = int((area==8).sum()) 
    if not _has_eight_pair(ctx):
        return 0.0
    return sigmoid(eight_count, k = 0.5, x0 = 3.0)

@registry.heuristic(weight=1.0)
def feature_remove_seven(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 7).any(): 
        return 0.0
    seven_count = int((area==7).sum()) 
    if not _has_seven_pair(ctx): 
        return 0.0
    return sigmoid(seven_count, k = 0.5, x0 = 3.0)

@registry.heuristic(weight=1.0)
def feature_remove_six(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 6).any():
        return 0.0
    six_count = int((area==6).sum())
    if not _has_six_pair(ctx):
        return 0.0
    return sigmoid(six_count, k = 0.5, x0 = 3.0) 

@registry.heuristic(weight=1.0)
def feature_remove_five(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 5).any():
        return 0.0
    five_count = int((area==5).sum()) 
    if not _has_five_pair(ctx):
        return 0.0
    return sigmoid(five_count, k = 0.5, x0 = 3.0)



def _has_nine_one_pair(ctx : FeatureContext) -> bool:
    for action in ctx.valid_actions:
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = ctx.board_array[r1:r2+1, c1:c2+1]
        if (region == 9).any():
            return True
    return False

def _has_eight_pair(ctx : FeatureContext) -> bool:
    for action in ctx.valid_actions:
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = ctx.board_array[r1:r2+1, c1:c2+1]
        if (region == 8).any():
            return True
    return False

def _has_seven_pair(ctx : FeatureContext) -> bool:
    for action in ctx.valid_actions:
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = ctx.board_array[r1:r2+1, c1:c2+1]
        if (region == 7).any():
            return True
    return False

def _has_six_pair(ctx : FeatureContext) -> bool:

    for action in ctx.valid_actions:
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = ctx.board_array[r1:r2+1, c1:c2+1]
        if (region == 6).any():
            return True
    return False

def _has_five_pair(ctx : FeatureContext) -> bool:
    for action in ctx.valid_actions:
        r1, c1 = action.top_left
        r2, c2 = action.bottom_right
        region = ctx.board_array[r1:r2+1, c1:c2+1]
        if (region == 5).any():
            return True
    return False
    
# def feature_remove_the_most_grouping(ctx: FeatureContext) -> float:
#     return float((ctx.area != 0).sum()) / ctx.area.size