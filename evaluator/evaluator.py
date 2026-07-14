from game.action import Action

from ..feature_assistance.feature_context import FeatureContext
from ..feature_assistance.feature_spec import FEATURES

# 각 feature_spec에 대해, feature_context를 받아서 점수를 계산하는 함수
def evaluate(ctx: FeatureContext, weights: dict[str, float]) -> float:
    score = 0.0
    for feature_spec in FEATURES:
        feature_func = feature_spec.func
        weight = weights.get(feature_spec.name, feature_spec.weight)
        score += weight * feature_func(ctx)
    return score


# 주어진 보드 상태에서 가능한 모든 액션 중, 가장 높은 점수를 가진 액션을 선택하는 함수
def pick_best_action(actions, board, weights: dict[str, float]) -> Action:
    best_action = None
    best_score = float('-inf')

    for action in actions:
        ctx = FeatureContext.from_board(board, action)
        score = evaluate(ctx, weights)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action

