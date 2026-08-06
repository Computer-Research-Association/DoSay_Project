"""공유 보드 유틸 — 모든 모델(greedy/beam/anneal/mcts)이 이 파일을 씀."""
from __future__ import annotations
import math
import numpy as np

ROWS, COLS = 9, 18
TOTAL = ROWS * COLS


def make_board(seed: int) -> np.ndarray:
    """시드로 9×18 보드 (값 1-9)."""
    return np.random.default_rng(seed).integers(1, 10, size=(ROWS, COLS), dtype=np.int8)


def _prefix(grid: np.ndarray) -> np.ndarray:
    """2D 누적합 → 임의 사각형 합을 O(1)."""
    P = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    P[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    return P


def valid_actions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """합=10인 '최소 사각형' 유효 수 (테두리 4변이 안 빈 것)."""
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


def apply_move(grid: np.ndarray, mv) -> int:
    """mv 사각형의 남은 사과 제거 (in-place). 제거 칸 수 반환."""
    r1, c1, r2, c2 = mv
    region = grid[r1:r2 + 1, c1:c2 + 1]
    cleared = int(np.count_nonzero(region))
    region[:] = 0
    return cleared


def cells_of(grid: np.ndarray, mv) -> int:
    r1, c1, r2, c2 = mv
    return int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))


def sigmoid(x, k, x0):
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def heuristic(grid: np.ndarray) -> float:
    """보드 평가 = nine(9/1 짝) + eight(8/2 짝) + action_count.
    greedy/beam 이 이 값을 최대화. (feature engineering으로 찾은 best 조합)"""
    n9 = int((grid == 9).sum()); n1 = int((grid == 1).sum())
    n8 = int((grid == 8).sum()); n2 = int((grid == 2).sum())
    f9 = 0.0 if n9 == 0 else (-1.0 if n1 == 0 else -sigmoid(n9 / n1, 2.5, 1.0))
    f8 = 0.0 if n8 == 0 else (-1.0 if n2 == 0 else -sigmoid(n8 / n2, 2.5, 1.0))
    fa = sigmoid(len(valid_actions(grid)), 0.2, 15.0)
    return f9 + f8 + fa
