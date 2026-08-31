# 정책: 큰 수 우선(region 최대값 큰 순) + 적게 지우기
import numpy as np

def rollout_policy(grid, actions, rng):
    best, best_key = None, None
    for a in actions:
        r1, c1, r2, c2 = a
        region = grid[r1:r2+1, c1:c2+1]
        cells = int(np.count_nonzero(region))
        maxv = int(region.max())
        key = (maxv, -cells)
        if best_key is None or key > best_key:
            best_key, best = key, a
    return best
