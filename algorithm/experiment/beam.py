"""
빔서치 실험 (팀원 executor 안 건드리고 여기서 프로토타입).

- 모델(version .py)의 registry(heuristic 가중합)를 그대로 사용, 탐색 깊이만 추가.
- 한 수 결정: 현재 보드에서 depth수 앞을 보되, 각 단계마다 heuristic 상위 width개만 유지(beam).
  최종 최고 리프의 '첫 수'를 실제로 둔다.
- width=1, depth=1 이면 greedy와 동일(정확성 검증용).
- 복사는 gridcopy(Board.from_board(grid.copy())) — deepcopy보다 가벼움.

사용법:
    python algorithm/experiment/beam.py --version Greedy_V08c --width 2 --depth 2 --games 100
"""
import os
import sys
import json
import time
import argparse
import importlib
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
os.chdir(ROOT)

from game.board import Board          # noqa: E402

GRID = (9, 18)
TOTAL = GRID[0] * GRID[1]
RESULTS_DIR = ROOT / "results"
BASE_SEED = 1234

_REGISTRY = None

def _init(version: str, weights: dict | None = None):
    global _REGISTRY
    m = importlib.import_module(f"algorithm.models.version.{version}")
    _REGISTRY = m.registry
    if weights:                       # 지정된 weight로 덮어쓰기 (튜닝용)
        for e in _REGISTRY.entries:
            if e.name in weights:
                e.weight = weights[e.name]

def _apply(board, action):
    """action을 둔 새 보드 + 지운 칸 수 반환 (gridcopy)."""
    g = board.grid.copy()
    (r1, c1), (r2, c2) = action.top_left, action.bottom_right
    cleared = int(np.count_nonzero(g[r1:r2+1, c1:c2+1]))
    g[r1:r2+1, c1:c2+1] = 0
    return Board.from_board(g), cleared

def _beam_choose(board, width, depth, alpha):
    """리프 점수 = heuristic(leaf) + alpha * (누적제거/TOTAL). 최고 리프의 '첫 수' 반환."""
    def value(h, cum):
        return h + alpha * (cum / TOTAL)

    root = board.get_valid_actions()
    if not root:
        return None
    # 1단계: 루트의 모든 수 전개
    beam = []   # (board, first_action, cum_cleared, h_score)
    for a in root:
        child, cl = _apply(board, a)
        beam.append((child, a, cl, _REGISTRY.evaluate(child)))
    beam.sort(key=lambda x: value(x[3], x[2]), reverse=True)
    beam = beam[:width]
    # 2..depth 단계: beam만 더 전개
    for _ in range(depth - 1):
        cand = []
        for b, fa, cum, h in beam:
            acts = b.get_valid_actions()
            if not acts:
                cand.append((b, fa, cum, h))       # 끝난 경로는 유지
                continue
            for a in acts:
                child, cl = _apply(b, a)
                cand.append((child, fa, cum + cl, _REGISTRY.evaluate(child)))
        cand.sort(key=lambda x: value(x[3], x[2]), reverse=True)
        beam = cand[:width]
    return max(beam, key=lambda x: value(x[3], x[2]))[1]

def _play(arg):
    seed, width, depth, alpha = arg
    board = Board.from_seed(GRID, seed)
    score = 0
    while True:
        if not board.get_valid_actions():
            break
        a = _beam_choose(board, width, depth, alpha)
        if a is None:
            break
        (r1, c1), (r2, c2) = a.top_left, a.bottom_right
        score += int(np.count_nonzero(board.grid[r1:r2+1, c1:c2+1]))
        board.do_action(a)
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="Greedy_V08c")
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.0)   # 누적제거 반영 강도 (0=순수 heuristic)
    p.add_argument("--weights", type=str, default=None)  # JSON dict: 피처 weight 오버라이드
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--quiet", action="store_true")       # avg만 출력 (튜닝용)
    args = p.parse_args()

    weights = json.loads(args.weights) if args.weights else None
    seeds = [(BASE_SEED + i, args.width, args.depth, args.alpha) for i in range(args.games)]
    n_workers = min(os.cpu_count() or 1, args.games)

    if not args.quiet:
        print(f"[{args.version}] beam width={args.width} depth={args.depth} alpha={args.alpha}, {args.games}판 실행 중...")
    t0 = time.perf_counter()
    with Pool(processes=n_workers, initializer=_init, initargs=(args.version, weights)) as pool:
        scores = pool.map(_play, seeds)
    dt = time.perf_counter() - t0

    arr = np.array(scores)
    if args.quiet:                      # 튜닝용: avg만 출력, 파일 저장 안 함
        print(f"{arr.mean():.4f}")
        return
    print(f"\n=== {args.version}  beam(w={args.width}, d={args.depth}) ===")
    print(f" avg score  : {arr.mean():.2f}  (ratio {arr.mean()/TOTAL:.3f})")
    print(f" std        : {arr.std():.2f}")
    print(f" min / max  : {arr.min()} / {arr.max()}")
    print(f" time       : {dt:.0f}s  ({dt/args.games:.2f}s/게임)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.version}_beam_w{args.width}_d{args.depth}_a{args.alpha}.json"
    out.write_text(json.dumps({
        "version": args.version, "beam_width": args.width, "max_depth": args.depth,
        "alpha": args.alpha, "n_games": args.games, "base_seed": BASE_SEED,
        "avg_score": round(float(arr.mean()), 2),
        "std_score": round(float(arr.std()), 2),
        "avg_ratio": round(float(arr.mean())/TOTAL, 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" → 저장: {out.name}")


if __name__ == "__main__":
    main()
