"""
MCTS (Monte Carlo Tree Search) — 어려운/중간 판 독립 검증용.

- 결정적 단일 플레이어 최대화 → best-value 백업(평균 아님).
- 트리: 노드 = grid 상태. 선택=UCT, 확장=미시도 액션 1개, 시뮬=rollout 정책(worst/fewest), 역전파=max.
- 최종 = 모든 rollout 중 최고 종료 점수 (결정적이라 global best가 답).
- anneal.py의 grid 유틸/정책 재사용.

사용법:
    python algorithm/experiment/mcts.py --version Anneal_V01b --seed 1236 --iters 20000
    python algorithm/experiment/mcts.py --version Anneal_V01b --iters 20000 --games 20
"""
import os, sys, math, time, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

from algorithm.experiment.anneal import (   # noqa: E402
    _make_grid, _valid_actions, _apply, _load_policy, _POLICY, TOTAL, BASE_SEED,
)
import algorithm.experiment.anneal as A     # _POLICY 최신 참조용

C_UCT = 0.7   # 탐색 상수


class Node:
    __slots__ = ("grid", "score", "untried", "children", "N", "Q")
    def __init__(self, grid, score):
        self.grid = grid           # 이 노드 상태
        self.score = score         # 여기까지 제거 칸(루트=0)
        self.untried = _valid_actions(grid)   # 미확장 액션
        self.children = []         # [(action, Node)]
        self.N = 0
        self.Q = score             # 서브트리 최고 종료 점수 (best 백업)


def _rollout_from(grid, rng, greedy_prob):
    """grid에서 정책대로 끝까지. 추가 제거 칸 반환. grid 소모."""
    policy = A._POLICY
    total = 0
    while True:
        acts = _valid_actions(grid)
        if not acts:
            return total
        if rng.random() >= greedy_prob:
            a = acts[rng.randrange(len(acts))]
        else:
            a = policy(grid, acts, rng)
        total += _apply(grid, a)


def _uct_child(node):
    logN = math.log(node.N + 1)
    best, best_v = None, -1.0
    for a, ch in node.children:
        exploit = ch.Q / TOTAL                       # 정규화된 best value
        explore = C_UCT * math.sqrt(logN / (ch.N + 1e-9))
        v = exploit + explore
        if v > best_v:
            best_v, best = v, (a, ch)
    return best


def mcts_one(seed, iters, rng, greedy_prob=0.8):
    root = Node(_make_grid(seed), 0)
    global_best = 0
    for _ in range(iters):
        node = root
        path = [root]
        # 선택: 완전 확장된 노드는 UCT로 내려감
        while not node.untried and node.children:
            _, node = _uct_child(node)
            path.append(node)
        # 확장
        if node.untried:
            a = node.untried.pop(rng.randrange(len(node.untried)))
            child_grid = node.grid.copy()
            cleared = _apply(child_grid, a)
            child = Node(child_grid, node.score + cleared)
            node.children.append((a, child))
            path.append(child)
            node = child
        # 시뮬레이션
        add = _rollout_from(node.grid.copy(), rng, greedy_prob)
        leaf = node.score + add
        if leaf > global_best:
            global_best = leaf
        # 역전파 (max 백업)
        for nd in path:
            nd.N += 1
            if leaf > nd.Q:
                nd.Q = leaf
    return global_best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="Anneal_V01b")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--greedy-prob", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _load_policy(args.version)

    if args.seed is not None:
        rng = random.Random(args.seed * 1000 + args.instance)
        print(mcts_one(args.seed, args.iters, rng, args.greedy_prob))
        return

    t0 = time.perf_counter()
    scores = [mcts_one(BASE_SEED + i, args.iters, random.Random(BASE_SEED + i), args.greedy_prob)
              for i in range(args.games)]
    dt = time.perf_counter() - t0
    arr = np.array(scores)
    if args.quiet:
        print(f"{arr.mean():.4f}"); return
    print(f"MCTS iters={args.iters} gp={args.greedy_prob}, {args.games}판")
    print(f" avg {arr.mean():.2f}  min {arr.min()} max {arr.max()}  {dt/args.games:.1f}s/판")


if __name__ == "__main__":
    main()
