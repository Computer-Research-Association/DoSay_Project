# 정책: 적게 지우기만 (9 우선 없음)
import numpy as np

def rollout_policy(grid, actions, rng):
    best, best_cells = None, None
    for a in actions:
        r1, c1, r2, c2 = a
        cells = int(np.count_nonzero(grid[r1:r2+1, c1:c2+1]))
        if best_cells is None or cells < best_cells:
            best_cells, best = cells, a
    return best
