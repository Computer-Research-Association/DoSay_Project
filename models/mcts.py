"""
MCTS (Monte Carlo Tree Search) — best-value 백업 (결정적 단일 플레이어 최대화).
  선택=UCB1, 확장=미시도 수 1개, 시뮬=적게지우기 rollout, 역전파=경로 최고점.
  최종 = 모든 rollout 중 최고 종료 점수.

성능: 어닐링보다 낮고(하드 판 116 vs 113) 2배 느림. 이 게임엔 부적합.
      (전이에 무작위성이 없어 MCTS 강점이 안 살고, rollout이 무거움)

사용: python -m models.mcts --seed 1234 --iters 3000
"""
import math
import random
import argparse
import numpy as np
from models.board import make_board, valid_actions, apply_move, cells_of, TOTAL

C_UCT = 0.7


def _rollout(grid, rng):
    """적게지우기 위주 rollout. 추가 제거 칸 반환. grid 소모."""
    total = 0
    while True:
        acts = valid_actions(grid)
        if not acts:
            return total
        a = min(acts, key=lambda m: cells_of(grid, m)) if rng.random() < 0.8 else acts[rng.randrange(len(acts))]
        total += apply_move(grid, a)


class Node:
    __slots__ = ("grid", "score", "moves", "untried", "children", "N", "Q")

    def __init__(self, grid, score):
        self.grid = grid
        self.score = score
        self.moves = valid_actions(grid)
        self.untried = list(self.moves)
        self.children = []          # (move, Node)
        self.N = 0
        self.Q = score              # 서브트리 최고 종료 점수


def _uct(node):
    logN = math.log(node.N + 1)
    return max(node.children, key=lambda mc: mc[1].Q / TOTAL + C_UCT * math.sqrt(logN / (mc[1].N + 1e-9)))


def mcts(board, iters, rng):
    root = Node(board.copy(), 0)
    best = 0
    for _ in range(iters):
        node, path = root, [root]
        while not node.untried and node.children:      # 선택
            node = _uct(node)[1]; path.append(node)
        if node.untried:                               # 확장
            a = node.untried.pop(rng.randrange(len(node.untried)))
            cg = node.grid.copy(); cl = apply_move(cg, a)
            child = Node(cg, node.score + cl)
            node.children.append((a, child)); path.append(child); node = child
        leaf = node.score + _rollout(node.grid.copy(), rng)   # 시뮬
        best = max(best, leaf)
        for nd in path:                                # 역전파 (max)
            nd.N += 1
            if leaf > nd.Q:
                nd.Q = leaf
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--iters", type=int, default=3000)
    args = p.parse_args()
    print(f"[mcts {args.iters}] seed {args.seed}: {mcts(make_board(args.seed), args.iters, random.Random(0))}/{TOTAL}")


if __name__ == "__main__":
    main()
