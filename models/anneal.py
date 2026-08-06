"""
Simulated Annealing (worst-이웃) + 병렬 max — ★ BEST 모델.
  해 = 한 판의 액션 시퀀스.  이웃 = 'worst' 지점을 다른 수로 갈아타고 뒤를 재플레이.
  rollout 정책 = 적게 지우기(딱 맞는 짝).  수락 = Metropolis(온도 냉각).
  병렬 = 같은 보드 여러 인스턴스(rng만 다름) → 최고 선택.

성능: 랜덤 100판 평균 ~135 (max-of-10, 2800 iter, 판당 ~70초).
      판별 near-최적 (추정 천장 대비 -2). 모든 시도(그리디·빔·MCTS·DL) 중 최고.
      iters·instances 키우면 천장(~137.8) 근접.

사용:
    from models.anneal import deploy
    score, seq = deploy(make_board(1234), iters=2800, instances=10)
    python -m models.anneal --seed 1234 --iters 2800 --instances 10
"""
import math
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from models.board import make_board, valid_actions, apply_move, cells_of, TOTAL


def _fewest_cells(grid, actions):
    """rollout 정책: 적게 지우는(딱 맞는 짝) 수. 여러 정책 실험 중 최선이었음."""
    best, best_cells = actions[0], 1 << 30
    for a in actions:
        c = cells_of(grid, a)
        if c < best_cells:
            best_cells, best = c, a
    return best


def _rollout(grid, first, rng, greedy_prob):
    """first 두고 끝까지 플레이. (시퀀스, 총제거). grid 소모."""
    seq, total, a = [], 0, first
    while a is not None:
        total += apply_move(grid, a)
        seq.append(a)
        acts = valid_actions(grid)
        if not acts:
            break
        a = acts[rng.randrange(len(acts))] if rng.random() >= greedy_prob else _fewest_cells(grid, acts)
    return seq, total


def _build_prefix(board, actions):
    grid = board.copy(); pg, pc, acc = [grid.copy()], [0], 0
    for a in actions:
        acc += apply_move(grid, a); pg.append(grid.copy()); pc.append(acc)
    return pg, pc


def _pick_worst(n, pc, rng):
    """worst 이웃 지점: 딱 맞는 짝(2칸) 초과로 낭비한 수일수록 우선 수정."""
    w = [(pc[i + 1] - pc[i] - 2) ** 2 + 0.1 for i in range(n)]
    r = rng.random() * sum(w); acc = 0.0
    for i, wi in enumerate(w):
        acc += wi
        if acc >= r:
            return i
    return n - 1


def anneal_once(board, iters=2800, rng_seed=0, T0=3.0, greedy_prob=0.8):
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
        base_g = pg[t]; acts = valid_actions(base_g)
        if len(acts) < 2:
            continue
        alt = [a for a in acts if a != cur_seq[t]]
        if not alt:
            continue
        tail_seq, tail_total = _rollout(base_g.copy(), alt[rng.randrange(len(alt))], rng, greedy_prob)
        new_score = pc[t] + tail_total
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / T):
            cur_seq, cur_score = cur_seq[:t] + tail_seq, new_score
            pg, pc = _build_prefix(board, cur_seq)
            if cur_score > best_score:
                best_seq, best_score = cur_seq, cur_score
    return best_score, best_seq


def _worker(arg):
    return anneal_once(*arg)


def deploy(board, iters=2800, instances=10, T0=3.0, greedy_prob=0.8):
    """병렬 max-of-instances. (score, sequence) 반환. spawn으로 macOS 데드락 회피."""
    args = [(board, iters, i, T0, greedy_prob) for i in range(instances)]
    if instances == 1:
        return _worker(args[0])
    with ProcessPoolExecutor(max_workers=instances, mp_context=mp.get_context("spawn")) as ex:
        return max(ex.map(_worker, args), key=lambda r: r[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--iters", type=int, default=2800)
    p.add_argument("--instances", type=int, default=10)
    args = p.parse_args()
    board = make_board(args.seed)
    score, seq = deploy(board, iters=args.iters, instances=args.instances)
    print(f"[anneal] seed {args.seed}: {score}/{TOTAL}  ({len(seq)}수)")


if __name__ == "__main__":
    main()
