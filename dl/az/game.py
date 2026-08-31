"""
사과게임 환경 (AlphaZero용, grid 레벨).
- 상태 = np.int8 (9,18), 0=빈칸.
- 합법수 = 합 10인 '최소 사각형' (r1,c1,r2,c2).  board.py 규칙과 동일.
- 적용 시 제거 칸 수 반환.
"""
import numpy as np

ROWS, COLS = 9, 18
TOTAL = ROWS * COLS


def make_board(seed):
    return np.random.default_rng(seed).integers(1, 10, size=(ROWS, COLS), dtype=np.int8)


def _prefix(grid):
    P = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    P[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    return P


def legal_moves(grid):
    """합=10 최소 사각형 목록 [(r1,c1,r2,c2), ...]."""
    P = _prefix(grid)

    def area(r1, c1, r2, c2):
        return int(P[r2 + 1, c2 + 1] - P[r1, c2 + 1] - P[r2 + 1, c1] + P[r1, c1])

    res = []
    for r1 in range(ROWS):
        for r2 in range(r1, ROWS):
            for c1 in range(COLS):
                for c2 in range(c1, COLS):
                    s = area(r1, c1, r2, c2)
                    if s == 10:
                        if area(r1, c1, r1, c2) == 0: continue
                        if area(r1, c2, r2, c2) == 0: continue
                        if area(r2, c1, r2, c2) == 0: continue
                        if area(r1, c1, r2, c1) == 0: continue
                        res.append((r1, c1, r2, c2))
                    elif s > 10:
                        break
    return res


def apply_move(grid, mv):
    """mv 적용 (in-place). 제거 칸 수 반환."""
    r1, c1, r2, c2 = mv
    region = grid[r1:r2 + 1, c1:c2 + 1]
    cleared = int(np.count_nonzero(region))
    region[:] = 0
    return cleared
