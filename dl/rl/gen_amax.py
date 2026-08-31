"""
③ 데이터 — (상태, action-max가 고른 수) 지도학습 라벨 생성.
action-max = 두고 나서 남는 유효수가 최대인 수 (우리 최고 그리디, 벡터화 카운트).
저장: dl/rl/amax_data.pkl  [(grid, moves, chosen_idx), ...]
"""
import os, sys, pickle, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dl.az.game import make_board, legal_moves, apply_move            # noqa: E402
from algorithm.models.version.Anneal_V02action import _count_valid    # noqa: E402


def amax_choice(grid, moves):
    best_i, best_n = 0, -1
    for i, (r1, c1, r2, c2) in enumerate(moves):
        region = grid[r1:r2 + 1, c1:c2 + 1]; saved = region.copy(); region[:] = 0
        n = _count_valid(grid); region[:] = saved
        if n > best_n:
            best_n, best_i = n, i
    return best_i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--base", type=int, default=60000)
    p.add_argument("--sample", type=float, default=0.5)
    args = p.parse_args()
    rng = random.Random(0)
    data = []
    for gi in range(args.games):
        g = make_board(args.base + gi)
        while True:
            mv = legal_moves(g)
            if not mv:
                break
            if rng.random() < args.sample and len(mv) >= 2:
                data.append((g.copy(), mv, amax_choice(g, mv)))
            # 다양화: 대부분 action-max 따라가되 가끔 랜덤
            if rng.random() < 0.3:
                i = rng.randrange(len(mv))
            else:
                i = amax_choice(g, mv)
            apply_move(g, mv[i])
    with open(ROOT / "dl" / "rl" / "amax_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"샘플 {len(data)}  ({args.games}판)  → dl/rl/amax_data.pkl")


if __name__ == "__main__":
    main()
