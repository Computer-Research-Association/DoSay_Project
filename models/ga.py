"""
Genetic Algorithm — 개인=수순, 적합도=제거 칸수.
  - 교차(crossover): 부모A 앞부분 + 부모B의 아직 유효한 수 이식 + rollout으로 마무리
  - 변이(mutation): 한 지점부터 다시 rollout (어닐링 이웃과 유사)
  - 선택=토너먼트, 엘리트 보존.

어닐링(SA)과 같은 블랙박스 계열. 이 게임 천장(~137.8) 안에서 SA와 비교용.

사용: python -m models.ga --seed 1234 --pop 40 --gens 70
"""
import argparse
import random
import numpy as np
from models.board import make_board, valid_actions, apply_move, cells_of, TOTAL


def _fewest(grid, acts):
    return min(acts, key=lambda a: cells_of(grid, a))


def _rollout(grid, rng, gp):
    seq, total = [], 0
    while True:
        acts = valid_actions(grid)
        if not acts:
            return seq, total
        a = acts[rng.randrange(len(acts))] if rng.random() >= gp else _fewest(grid, acts)
        total += apply_move(grid, a); seq.append(a)


def make_ind(board, rng, gp=0.7):
    return _rollout(board.copy(), rng, gp)          # (seq, score)


def crossover(board, A, B, rng):
    """A 앞부분 + B의 아직 유효한 수 이식 + rollout 마무리."""
    k = rng.randint(1, max(1, len(A) - 1))
    g = board.copy(); child, score = [], 0
    for a in A[:k]:
        score += apply_move(g, a); child.append(a)
    for b in B:                                     # B의 수 중 지금도 유효한 것만 이식
        r1, c1, r2, c2 = b
        reg = g[r1:r2 + 1, c1:c2 + 1]
        if int(reg.sum()) == 10 and int(np.count_nonzero(reg)) > 0:
            score += apply_move(g, b); child.append(b)
    ts, tt = _rollout(g, rng, 0.8)
    return child + ts, score + tt


def mutate(board, seq, rng):
    if len(seq) < 2:
        return seq, sum(cells_of(board, a) for a in seq)  # 근사; 거의 안 쓰임
    t = rng.randrange(len(seq))
    g = board.copy(); score = 0
    for a in seq[:t]:
        score += apply_move(g, a)
    ts, tt = _rollout(g, rng, 0.8)
    return seq[:t] + ts, score + tt


def _tournament(pop, rng, k=3):
    return max((pop[rng.randrange(len(pop))] for _ in range(k)), key=lambda x: x[1])


def ga(board, pop=40, gens=70, rng=None, elite_frac=0.2, mut_prob=0.4):
    rng = rng or random.Random(0)
    P = [make_ind(board, rng) for _ in range(pop)]
    best = max(P, key=lambda x: x[1])
    n_elite = max(1, int(pop * elite_frac))
    for _ in range(gens):
        P.sort(key=lambda x: -x[1])
        newP = P[:n_elite]                          # 엘리트 보존
        while len(newP) < pop:
            A = _tournament(P, rng); B = _tournament(P, rng)
            child = crossover(board, A[0], B[0], rng)
            if rng.random() < mut_prob:
                child = mutate(board, child[0], rng)
            newP.append(child)
        P = newP
        gb = max(P, key=lambda x: x[1])
        if gb[1] > best[1]:
            best = gb
    return best[1], best[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--gens", type=int, default=70)
    args = p.parse_args()
    sc, seq = ga(make_board(args.seed), pop=args.pop, gens=args.gens, rng=random.Random(0))
    print(f"[GA pop{args.pop} gens{args.gens}] seed {args.seed}: {sc}/{TOTAL}  ({len(seq)}수)")


if __name__ == "__main__":
    main()
