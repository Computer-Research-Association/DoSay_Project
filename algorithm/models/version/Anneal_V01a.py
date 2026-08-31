# 정책: 9 우선 + 적게 지우기 (현재 기준)
import numpy as np

def rollout_policy(grid, actions, rng):
    best, best_key = None, None
    for a in actions:
        r1, c1, r2, c2 = a
        region = grid[r1:r2+1, c1:c2+1]
        cells = int(np.count_nonzero(region))
        has9 = bool((region == 9).any())
        key = (1 if has9 else 0, -cells)
        if best_key is None or key > best_key:
            best_key, best = key, a
    return best
