from .feature_context import FeatureContext
from .math_algorithm import sigmoid

#숫자 편향 features ex) 지울 수 있는 'n'이라는 숫자가 몇 개 남았는가 점수를 올림으로써 지우게끔 유도
def feature_remove_nine(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 9).any(): #격자 안에 9가 없을시 0을 return한다
        return 0.0
    nine_count = int((area==9).sum()) #area안에 9가 있는 수의 개수
    if not _has_nine_one_pair(ctx): #9와 짝 지어지는 경우의 수가 판 내에 존재하는지
        return 0.0
    return sigmoid(nine_count, k = 0.5, x0 = 3.0) #추후 이 값을 조정하면서 k 값과 x0 값을 찾아도 됨

def feature_remove_eight(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 8).any():
        return 0.0
    eight_count = int((area==8).sum()) 
    if not _has_eight_pair(ctx):
        return 0.0
    return sigmoid(eight_count, k = 0.5, x0 = 3.0)


def feature_remove_seven(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 7).any(): 
        return 0.0
    seven_count = int((area==7).sum()) 
    if not _has_seven_pair(ctx): 
        return 0.0
    return sigmoid(seven_count, k = 0.5, x0 = 3.0)

def feature_remove_six(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 6).any():
        return 0.0
    six_count = int((area==6).sum())
    if not _has_six_pair(ctx):
        return 0.0
    return sigmoid(six_count, k = 0.5, x0 = 3.0) 

def feature_remove_five(ctx: FeatureContext) -> float:
    area = ctx.board_array
    if not(area == 5).any():
        return 0.0
    five_count = int((area==5).sum()) 
    if not _has_five_pair(ctx):
        return 0.0
    return sigmoid(five_count, k = 0.5, x0 = 3.0)


#feature_remove series를 구현하기 위한 보조 함수
#9와 짝이 되는 valid action이 있는지 여부 확인
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
    
def feature_remove_the_most_grouping(ctx : FeatureContext) -> float:
    (x1, y1), (x2, y2) = ctx.action.top_left, ctx.action.bottom_right
    area = ctx.board_array[x1:x2+1, y1:y2+1]
    count = (area != 0).sum()

    return float(count) / 10.0




