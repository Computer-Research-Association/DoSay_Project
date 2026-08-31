"""
Stochastic Beam Search — 빔서치에 랜덤성 주입 (grid 레벨 경량).

heuristic = nine(9/1) + eight(8/2) + action_count  (Beam_V01a와 동일, 결정적 beam=118.16 기준)

모드 3가지:
  stochastic : 각 beam 단계에서 상위 K를 결정적으로 안 고르고 softmax(value/T) 확률로 샘플.
               T는 초반(보드 꽉 참)에 높고 후반에 낮아짐 → "초반 랜덤성". 단일 실행.
  restart    : stochastic 게임을 R번 돌려 최고 점수 선택 (랜덤성으로 다양성 확보).
  randprefix : 초반 N수 순수 랜덤 → 이후 결정적 beam. R번 반복, 최고 선택.

사용법:
    python algorithm/experiment/sbeam.py --mode restart --seed 1234 --width 5 --depth 3 --temp 0.3 --restarts 4
"""
import os
import sys
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from game.board import compute_prefix_sum   # noqa: E402

GRID = (9, 18)
TOTAL = GRID[0] * GRID[1]
BASE_SEED = 1234


def _sigmoid(x, k, x0):
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

def _make_grid(seed):
    return np.random.default_rng(seed).integers(1, 10, size=GRID, dtype=np.int8)

def _valid_actions(grid):
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
                        res.append((r1, c1, r2, c2))
                    elif s > 10:
                        break
    return res

def _apply(grid, a):
    r1, c1, r2, c2 = a
    g = grid.copy()
    cleared = int(np.count_nonzero(g[r1:r2+1, c1:c2+1]))
    g[r1:r2+1, c1:c2+1] = 0
    return g, cleared

def _heuristic(grid):
    """nine + eight + action_count (Beam_V01a와 동일)."""
    n9 = int((grid == 9).sum()); n1 = int((grid == 1).sum())
    n8 = int((grid == 8).sum()); n2 = int((grid == 2).sum())
    f9 = 0.0 if n9 == 0 else (-1.0 if n1 == 0 else -_sigmoid(n9 / n1, 2.5, 1.0))
    f8 = 0.0 if n8 == 0 else (-1.0 if n2 == 0 else -_sigmoid(n8 / n2, 2.5, 1.0))
    nv = len(_valid_actions(grid))
    fa = _sigmoid(nv, 0.2, 15.0)
    return f9 + f8 + fa

def _select(cands, width, temp, rng):
    """cands=[(grid, first_action, value)]. temp=0이면 상위 width 결정적, >0이면 softmax 샘플."""
    if temp <= 1e-9 or len(cands) <= width:
        cands.sort(key=lambda x: x[2], reverse=True)
        return cands[:width]
    vals = np.array([c[2] for c in cands])
    logits = (vals - vals.max()) / temp
    probs = np.exp(logits); probs /= probs.sum()
    idx = rng.choice(len(cands), size=min(width, len(cands)), replace=False, p=probs)
    return [cands[i] for i in idx]

def _beam_choose(grid, width, depth, temp, rng):
    root = _valid_actions(grid)
    if not root:
        return None
    beam = []
    for a in root:
        child, _ = _apply(grid, a)
        beam.append((child, a, _heuristic(child)))
    beam = _select(beam, width, temp, rng)
    for _ in range(depth - 1):
        cand = []
        for g, fa, _v in beam:
            acts = _valid_actions(g)
            if not acts:
                cand.append((g, fa, _heuristic(g))); continue
            for a in acts:
                child, _ = _apply(g, a)
                cand.append((child, fa, _heuristic(child)))
        beam = _select(cand, width, temp, rng)
    return max(beam, key=lambda x: x[2])[1]

def _play_game(seed, width, depth, temp_fn, rand_prefix, rng):
    """한 게임 플레이. temp_fn(remaining)→temp, rand_prefix=초반 랜덤 수순."""
    grid = _make_grid(seed)
    score = 0
    move = 0
    while True:
        acts = _valid_actions(grid)
        if not acts:
            break
        if move < rand_prefix:
            a = acts[int(rng.integers(len(acts)))]          # 초반 랜덤
        else:
            remaining = int(np.count_nonzero(grid))
            a = _beam_choose(grid, width, depth, temp_fn(remaining), rng)
            if a is None:
                break
        grid, cl = _apply(grid, a)
        score += cl
        move += 1
    return score


def run_one(seed, args, rng):
    # 온도 함수: 초반(보드 꽉 참) 높고 후반 낮음
    def temp_fn(remaining):
        return args.temp * (remaining / TOTAL)

    if args.mode == "stochastic":
        return _play_game(seed, args.width, args.depth, temp_fn, 0, rng)
    if args.mode == "restart":
        return max(_play_game(seed, args.width, args.depth, temp_fn, 0, rng)
                   for _ in range(args.restarts))
    if args.mode == "randprefix":
        return max(_play_game(seed, args.width, args.depth, lambda r: 0.0, args.prefix, rng)
                   for _ in range(args.restarts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="restart", choices=["stochastic", "restart", "randprefix"])
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.3)    # 초반 온도
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--prefix", type=int, default=5)       # randprefix: 초반 랜덤 수
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.seed is not None:
        rng = np.random.default_rng(args.seed * 7919)
        print(run_one(args.seed, args, rng))
        return

    t0 = time.perf_counter()
    scores = []
    for i in range(args.games):
        rng = np.random.default_rng((BASE_SEED + i) * 7919)
        scores.append(run_one(BASE_SEED + i, args, rng))
    dt = time.perf_counter() - t0
    arr = np.array(scores)
    if args.quiet:
        print(f"{arr.mean():.4f}"); return
    print(f"[{args.mode}] w{args.width} d{args.depth} temp{args.temp} R{args.restarts} prefix{args.prefix}, {args.games}판")
    print(f" avg {arr.mean():.2f}  ratio {arr.mean()/TOTAL:.3f}  (min {arr.min()} max {arr.max()})  {dt/args.games:.1f}s/판")


if __name__ == "__main__":
    main()
