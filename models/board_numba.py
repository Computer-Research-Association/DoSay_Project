"""numba 컴파일판 valid_actions — 파이썬 인터프리터 오버헤드 제거."""
import numpy as np
from numba import njit


@njit(cache=True)
def _valid_njit(grid):
    ROWS, COLS = grid.shape
    P = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    for r in range(ROWS):
        for c in range(COLS):
            P[r + 1, c + 1] = grid[r, c] + P[r, c + 1] + P[r + 1, c] - P[r, c]
    out = np.empty((512, 4), dtype=np.int64)
    n = 0
    for r1 in range(ROWS):
        for r2 in range(r1, ROWS):
            for c1 in range(COLS):
                for c2 in range(c1, COLS):
                    s = P[r2 + 1, c2 + 1] - P[r1, c2 + 1] - P[r2 + 1, c1] + P[r1, c1]
                    if s == 10:
                        top = P[r1 + 1, c2 + 1] - P[r1, c2 + 1] - P[r1 + 1, c1] + P[r1, c1]
                        bot = P[r2 + 1, c2 + 1] - P[r2, c2 + 1] - P[r2 + 1, c1] + P[r2, c1]
                        left = P[r2 + 1, c1 + 1] - P[r1, c1 + 1] - P[r2 + 1, c1] + P[r1, c1]
                        right = P[r2 + 1, c2 + 1] - P[r1, c2 + 1] - P[r2 + 1, c2] + P[r1, c2]
                        if top > 0 and bot > 0 and left > 0 and right > 0:
                            out[n, 0] = r1; out[n, 1] = c1; out[n, 2] = r2; out[n, 3] = c2
                            n += 1
                    elif s > 10:
                        break
    return out[:n]


def valid_actions_numba(grid):
    arr = _valid_njit(grid)
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in arr]
