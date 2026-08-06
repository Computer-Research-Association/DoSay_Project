"""
Beam Search — 매 수, depth수 앞을 보되 각 단계 상위 width개만 유지(beam).
리프를 heuristic으로 평가 → 최고 리프의 '첫 수' 선택.  best: w5 d3, 평균 ~118-119.

한계: 리프를 '실제 최종점수'가 아니라 heuristic 근사로 평가 + 앞수 재검토 불가
      → 그리디 라인에 갇힘. 그래서 어닐링(135)에 크게 못 미침.

사용: python -m models.beam --seed 1234 --width 5 --depth 3
"""
import argparse
import numpy as np
from models.board import make_board, valid_actions, apply_move, heuristic, TOTAL


def _apply(grid, mv):
    r1, c1, r2, c2 = mv
    g = grid.copy(); cl = int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1])); g[r1:r2 + 1, c1:c2 + 1] = 0
    return g, cl


def beam_choose(grid, width, depth):
    root = valid_actions(grid)
    if not root:
        return None
    beam = []                                # (grid, first_action, heuristic)
    for a in root:
        child, _ = _apply(grid, a)
        beam.append((child, a, heuristic(child)))
    beam.sort(key=lambda x: x[2], reverse=True)
    beam = beam[:width]
    for _ in range(depth - 1):
        cand = []
        for g, fa, _h in beam:
            acts = valid_actions(g)
            if not acts:
                cand.append((g, fa, heuristic(g))); continue
            for a in acts:
                child, _ = _apply(g, a)
                cand.append((child, fa, heuristic(child)))
        cand.sort(key=lambda x: x[2], reverse=True)
        beam = cand[:width]
    return max(beam, key=lambda x: x[2])[1]


def play(board, width, depth):
    g = board.copy(); score = 0
    while True:
        if not valid_actions(g):
            return score
        a = beam_choose(g, width, depth)
        if a is None:
            return score
        score += apply_move(g, a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    args = p.parse_args()
    print(f"[beam w{args.width} d{args.depth}] seed {args.seed}: "
          f"{play(make_board(args.seed), args.width, args.depth)}/{TOTAL}")


if __name__ == "__main__":
    main()
