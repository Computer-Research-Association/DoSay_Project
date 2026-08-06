"""
Greedy — 매 수, '둔 뒤 보드 평가(heuristic)'가 가장 좋은 수를 선택.
best 휴리스틱: nine+eight+action_count.  성능: 랜덤 100판 평균 ~115.

한계: 한 수 앞만 봄(근시안). 장기 의존성(1을 아껴 9와 묶기 등)을 못 봄.

사용: python -m models.greedy --seed 1234
"""
import argparse
import numpy as np
from models.board import make_board, valid_actions, apply_move, heuristic, TOTAL


def play(board):
    g = board.copy(); score = 0
    while True:
        acts = valid_actions(g)
        if not acts:
            return score
        best, best_h = acts[0], -1e9
        for a in acts:                       # 각 수를 두면 보드가 얼마나 좋아지나
            r1, c1, r2, c2 = a
            child = g.copy(); child[r1:r2 + 1, c1:c2 + 1] = 0
            h = heuristic(child)
            if h > best_h:
                best_h, best = h, a
        score += apply_move(g, best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    print(f"[greedy] seed {args.seed}: {play(make_board(args.seed))}/{TOTAL}")


if __name__ == "__main__":
    main()
