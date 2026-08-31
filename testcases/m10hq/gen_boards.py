"""10의 배수 판 100개 생성 — 랜덤 생성 후 '마지막 칸'을 조정해 합≡0 mod10.
숫자 분포는 거의 그대로(162칸 중 1칸만 nudge). boards.npy 저장."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.board import make_board, ROWS, COLS


def make_mult10(seed):
    g = make_board(seed).copy()
    S = int(g.sum())
    d = (-S) % 10                       # 총합에 더해야 할 양 (mod 10)
    if d == 0:
        return g
    r, c = ROWS - 1, COLS - 1           # 마지막 칸 우선
    v = int(g[r, c])
    if v + d <= 9:
        g[r, c] = v + d; return g
    if v - (10 - d) >= 1:               # +d 안되면 등가인 -(10-d)
        g[r, c] = v - (10 - d); return g
    for i in range(ROWS):               # 마지막 칸으로 안되면 아무 칸
        for j in range(COLS):
            if int(g[i, j]) + d <= 9:
                g[i, j] = int(g[i, j]) + d; return g
            if int(g[i, j]) - (10 - d) >= 1:
                g[i, j] = int(g[i, j]) - (10 - d); return g
    return g


def main():
    boards = []
    for s in range(300000, 300100):
        b = make_mult10(s)
        assert int(b.sum()) % 10 == 0 and b.min() >= 1 and b.max() <= 9
        boards.append(b)
    arr = np.stack(boards)
    outdir = ROOT / "testcases" / "m10hq"; outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "boards.npy", arr)
    print(f"{len(boards)}개 판 (전부 합≡0 mod10, 이론상 최대 162) → boards.npy")
    print("합 예시:", [int(b.sum()) for b in boards[:5]])


if __name__ == "__main__":
    main()
