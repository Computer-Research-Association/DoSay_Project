"""10의 배수 판 1000개 — 랜덤 생성 후 마지막 칸 조정으로 합≡0 mod10. boards.npy 저장."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.board import make_board, ROWS, COLS


def make_mult10(seed):
    g = make_board(seed).copy()
    S = int(g.sum())
    d = (-S) % 10
    if d == 0:
        return g
    r, c = ROWS - 1, COLS - 1
    v = int(g[r, c])
    if v + d <= 9:
        g[r, c] = v + d; return g
    if v - (10 - d) >= 1:
        g[r, c] = v - (10 - d); return g
    for i in range(ROWS):
        for j in range(COLS):
            if int(g[i, j]) + d <= 9:
                g[i, j] = int(g[i, j]) + d; return g
            if int(g[i, j]) - (10 - d) >= 1:
                g[i, j] = int(g[i, j]) - (10 - d); return g
    return g


def main():
    boards = []
    for s in range(500000, 501000):
        b = make_mult10(s)
        assert int(b.sum()) % 10 == 0 and b.min() >= 1 and b.max() <= 9
        boards.append(b)
    np.save(ROOT / "testcases" / "m10big" / "boards.npy", np.stack(boards))
    print(f"{len(boards)}개 판 (전부 합≡0 mod10) → boards.npy")


if __name__ == "__main__":
    main()
