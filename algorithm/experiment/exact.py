"""
어려운 판 exact 탐사 — Branch & Bound (개선해 사냥).

- 목표: 어닐링 최선(incumbent)을 넘는 해를 찾거나, 시간예산 안에 못 넘으면 그 증거 축적.
- 상한(UB) = 지금까지 제거 칸 + 남은 칸 수 (느슨하지만 valid). score+UB <= incumbent 면 가지치기.
- 수순 정렬: '적게 지우는(딱 맞는 짝)' 수 우선 → 깊은 해를 빨리 찾음.
- 시간 제한 도달 시 지금까지 최선 반환 (증명 아님).

사용법:
    python algorithm/experiment/exact.py --seed 1236 --incumbent 116 --limit 600
"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from game.board import compute_prefix_sum   # noqa: E402

GRID = (9, 18)


def make_grid(seed):
    return np.random.default_rng(seed).integers(1, 10, size=GRID, dtype=np.int8)

def valid_actions(grid):
    H, W = grid.shape
    P = compute_prefix_sum(grid)
    def area(r1, c1, r2, c2):
        return int(P[r2+1, c2+1] - P[r1, c2+1] - P[r2+1, c1] + P[r1, c1])
    res = []
    for r1 in range(H):
        for r2 in range(r1, H):
            for c1 in range(W):
                for c2 in range(c1, W):
                    s = area(r1, c1, r2, c2)
                    if s == 10:
                        if area(r1, c1, r1, c2) == 0: continue
                        if area(r1, c2, r2, c2) == 0: continue
                        if area(r2, c1, r2, c2) == 0: continue
                        if area(r1, c1, r2, c1) == 0: continue
                        cells = int(np.count_nonzero(grid[r1:r2+1, c1:c2+1]))
                        res.append((cells, (r1, c1, r2, c2)))
                    elif s > 10:
                        break
    return res

def apply(grid, a):
    r1, c1, r2, c2 = a
    region = grid[r1:r2+1, c1:c2+1]
    cleared = int(np.count_nonzero(region))
    region[:] = 0
    return cleared


def solve(seed, incumbent, limit):
    grid = make_grid(seed)
    total_cells = int(np.count_nonzero(grid))
    total_value = int(grid.sum())
    t0 = time.time()
    stats = {"nodes": 0, "best": incumbent, "best_seq": None, "timeout": False}

    def dfs(g, score):
        if time.time() - t0 > limit:
            stats["timeout"] = True
            return
        stats["nodes"] += 1
        remaining = int(np.count_nonzero(g))
        if score + remaining <= stats["best"]:   # UB 가지치기
            return
        acts = valid_actions(g)
        if not acts:
            if score > stats["best"]:
                stats["best"] = score
            return
        acts.sort(key=lambda x: x[0])   # 적게 지우는 수 우선 (깊은 해 빨리)
        for cells, a in acts:
            if stats["timeout"]:
                return
            child = g.copy()
            apply(child, a)
            dfs(child, score + cells)

    dfs(grid, 0)
    dt = time.time() - t0
    print(f"seed {seed}: cells={total_cells} value={total_value}")
    print(f"  incumbent(어닐링) = {incumbent}")
    print(f"  B&B best         = {stats['best']}  ({'개선!' if stats['best'] > incumbent else '동일/이하'})")
    print(f"  nodes            = {stats['nodes']:,}")
    print(f"  time             = {dt:.0f}s  ({'시간초과(증명X)' if stats['timeout'] else '완전탐색 종료 → 최적 증명!'})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1236)
    p.add_argument("--incumbent", type=int, default=116)
    p.add_argument("--limit", type=float, default=600)
    args = p.parse_args()
    solve(args.seed, args.incumbent, args.limit)


if __name__ == "__main__":
    main()
