# 정책: 순수 랜덤 (대조군)
def rollout_policy(grid, actions, rng):
    return actions[rng.randrange(len(actions))]
