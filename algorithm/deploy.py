"""
사과게임 배포 모델 (best) — worst-이웃 Simulated Annealing + 병렬 max.

성능 (실측):
  - 랜덤 보드 평균 ~135 (max-of-10, 2800 iter, 판당 ~70초)
  - 판별로 near-최적 (추정 천장 대비 -2 정도)
  - 여러 독립 방법(빔·MCTS·부분 브루트포스) 중 최고

구조:
  1) 해 = 한 판의 액션 시퀀스 (액션 = 합10 최소 사각형 (r1,c1,r2,c2))
  2) 이웃 = 'worst' 지점(작은 수를 낭비한 수)을 골라 다른 유효수로 갈아타 → 뒤를 재플레이
  3) rollout 정책 = 적게 지우기(딱 맞는 짝 우선)
  4) 수락 = Metropolis (온도 선형 냉각)
  5) 병렬 = 같은 보드를 여러 인스턴스(rng만 다름) 돌려 최고 선택

사용:
    from algorithm.deploy import make_board, deploy
    board = make_board(1234)
    score, sequence = deploy(board, iters=2800, instances=10)

    # 또는 CLI
    python -m algorithm.deploy --seed 1234 --iters 2800 --instances 10
"""
from __future__ import annotations
import math
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROWS, COLS = 9, 18
TOTAL = ROWS * COLS


# ────────────────────────── 보드 유틸 ──────────────────────────
def make_board(seed: int) -> np.ndarray:
    """시드로 9×18 보드 생성 (값 1-9)."""
    return np.random.default_rng(seed).integers(1, 10, size=(ROWS, COLS), dtype=np.int8)


def _prefix(grid: np.ndarray) -> np.ndarray:
    """2D 누적합 (임의 사각형 합을 O(1)에)."""
    P = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    P[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    return P


def valid_actions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """합=10인 '최소 사각형' 유효 수 목록 (테두리 4변이 비지 않은 것)."""
    P = _prefix(grid)

    def area(r1, c1, r2, c2):
        return int(P[r2 + 1, c2 + 1] - P[r1, c2 + 1] - P[r2 + 1, c1] + P[r1, c1])

    res = []
    for r1 in range(ROWS):
        for r2 in range(r1, ROWS):
            for c1 in range(COLS):
                for c2 in range(c1, COLS):
                    s = area(r1, c1, r2, c2)
                    if s == 10:
                        if area(r1, c1, r1, c2) == 0: continue   # 위 변
                        if area(r1, c2, r2, c2) == 0: continue   # 오른 변
                        if area(r2, c1, r2, c2) == 0: continue   # 아래 변
                        if area(r1, c1, r2, c1) == 0: continue   # 왼 변
                        res.append((r1, c1, r2, c2))
                    elif s > 10:
                        break   # c2 늘리면 합만 커짐 → 가지치기
    return res


def apply_move(grid: np.ndarray, mv) -> int:
    """mv 사각형의 남은 사과 제거 (in-place). 제거 칸 수 반환."""
    r1, c1, r2, c2 = mv
    region = grid[r1:r2 + 1, c1:c2 + 1]
    cleared = int(np.count_nonzero(region))
    region[:] = 0
    return cleared


# ────────────────────────── rollout 정책 ──────────────────────────
def _fewest_cells(grid: np.ndarray, actions):
    """적게 지우는(딱 맞는 짝) 수 선택 — 검증된 최선의 rollout 정책."""
    best, best_cells = actions[0], 1 << 30
    for a in actions:
        r1, c1, r2, c2 = a
        cells = int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))
        if cells < best_cells:
            best_cells, best = cells, a
    return best


def _rollout(grid, first_action, rng, greedy_prob):
    """first_action 두고 끝까지 플레이. (시퀀스, 총 제거) 반환. grid 소모됨."""
    seq, total = [], 0
    a = first_action
    while a is not None:
        total += apply_move(grid, a)
        seq.append(a)
        acts = valid_actions(grid)
        if not acts:
            break
        a = acts[rng.randrange(len(acts))] if rng.random() >= greedy_prob else _fewest_cells(grid, acts)
    return seq, total


def _build_prefix(board, actions):
    """각 시점 (보드 스냅샷, 누적 제거) 캐싱."""
    grid = board.copy()
    pgrid, pclear, acc = [grid.copy()], [0], 0
    for a in actions:
        acc += apply_move(grid, a)
        pgrid.append(grid.copy())
        pclear.append(acc)
    return pgrid, pclear


def _pick_worst(n, pc, rng):
    """worst 이웃 지점: 딱 맞는 짝(2칸) 초과로 '낭비'한 수일수록 우선 수정."""
    w = [(pc[i + 1] - pc[i] - 2) ** 2 + 0.1 for i in range(n)]
    tot = sum(w)
    r = rng.random() * tot
    acc = 0.0
    for i, wi in enumerate(w):
        acc += wi
        if acc >= r:
            return i
    return n - 1


# ────────────────────────── 어닐링 ──────────────────────────
def anneal_once(board, iters=2800, rng_seed=0, T0=3.0, greedy_prob=0.8):
    """단일 어닐링 실행. (best_score, best_sequence) 반환."""
    rng = random.Random(rng_seed)
    acts = valid_actions(board)
    if not acts:
        return 0, []
    cur_seq, cur_score = _rollout(board.copy(), _fewest_cells(board, acts), rng, 1.0)
    best_seq, best_score = cur_seq, cur_score
    pg, pc = _build_prefix(board, cur_seq)

    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-6
        if len(cur_seq) < 2:
            break
        t = _pick_worst(len(cur_seq), pc, rng)
        base_g = pg[t]
        acts = valid_actions(base_g)
        if len(acts) < 2:
            continue
        cur_a = cur_seq[t]
        alt = [a for a in acts if a != cur_a]
        if not alt:
            continue
        nf = alt[rng.randrange(len(alt))]
        tail_seq, tail_total = _rollout(base_g.copy(), nf, rng, greedy_prob)
        new_score = pc[t] + tail_total
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            cur_seq = cur_seq[:t] + tail_seq
            cur_score = new_score
            pg, pc = _build_prefix(board, cur_seq)
            if cur_score > best_score:
                best_seq, best_score = cur_seq, cur_score
    return best_score, best_seq


def _worker(arg):
    board, iters, rng_seed, T0, gp = arg
    return anneal_once(board, iters, rng_seed, T0, gp)


def deploy(board, iters=2800, instances=10, T0=3.0, greedy_prob=0.8):
    """배포 모델: instances개 어닐링(같은 보드, rng만 다름) 병렬 → 최고 선택.
    (score, sequence) 반환. 병렬은 spawn 컨텍스트로 macOS fork+numpy 데드락 회피."""
    args = [(board, iters, i, T0, greedy_prob) for i in range(instances)]
    if instances == 1:
        return _worker(args[0])
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=instances, mp_context=ctx) as ex:
        results = list(ex.map(_worker, args))
    return max(results, key=lambda r: r[0])


# ────────────────────────── CLI ──────────────────────────
def main():
    p = argparse.ArgumentParser(description="사과게임 배포 모델 (worst-SA + 병렬 max)")
    p.add_argument("--seed", type=int, default=1234, help="보드 시드")
    p.add_argument("--iters", type=int, default=2800)
    p.add_argument("--instances", type=int, default=10)
    p.add_argument("--show-seq", action="store_true", help="수순 출력")
    args = p.parse_args()

    board = make_board(args.seed)
    score, seq = deploy(board, iters=args.iters, instances=args.instances)
    left = TOTAL - score
    print(f"seed {args.seed}: {score}/{TOTAL}  ({len(seq)}수, 남은 {left}칸)  "
          f"[합계 {int(board.sum())}, 이론상 최대 {162 if board.sum() % 10 == 0 else 161}]")
    if args.show_seq:
        for i, a in enumerate(seq, 1):
            print(f"  {i:2d}. {tuple(int(x) for x in a)}")


if __name__ == "__main__":
    main()
