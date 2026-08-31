"""
LAHC(Late Acceptance Hill Climbing) vs 어닐 — 이웃연산자 동일(worst-point), 수용전략만 다름.
LAHC: 온도 없이, L iter 전 값보다 낫거나 현재보다 낫거나면 수용. 이력 L만 튜닝.
사용: python lahc.py [budget초] [보드수]
"""
import sys, time, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.board import make_board, apply_move
from models.board_numba import valid_actions_numba as VA
from seq_compare import anneal as anneal_worst, _cells, _fewest, _rollout_tail, _pick_worst

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def neighbor(board, cur_S, rng, gp=0.8):
    gg = board.copy(); pcum = [0]
    for mv in cur_S: pcum.append(pcum[-1] + apply_move(gg, mv))
    j = _pick_worst(pcum, rng)
    g = board.copy(); cleared = 0
    for t in range(j): cleared += apply_move(g, cur_S[t])
    a = VA(g)
    if not a: return None
    choices = [m for m in a if m != cur_S[j]] or a
    mv = choices[rng.randrange(len(choices))]
    cleared += apply_move(g, mv)
    tc, ts = _rollout_tail(g, rng, gp)
    return cleared + tc, cur_S[:j] + [mv] + ts


def lahc(board, seed, budget, L, gp=0.8):
    rng = random.Random(seed)
    acts = VA(board)
    if not acts: return 0
    g = board.copy(); c0 = apply_move(g, _fewest(board, acts))
    tc, ts = _rollout_tail(g, rng, gp)
    cur_s, cur_S = c0 + tc, [_fewest(board, acts)] + ts
    best = cur_s
    hist = [cur_s] * L
    i = 0; t0 = time.time()
    while time.time() - t0 < budget:
        nb = neighbor(board, cur_S, rng, gp)
        if nb is not None:
            new_s, new_S = nb
            v = i % L
            if new_s >= cur_s or new_s >= hist[v]:
                cur_s, cur_S = new_s, new_S
            hist[v] = cur_s
            if cur_s > best: best = cur_s
        i += 1
    return best


def main():
    VA(make_board(1))
    print(f"=== 9×18 {NB}판 · LAHC vs 어닐 동일시간 (budget={BUDGET}s) ===")
    boards = [(400000 + k, make_board(400000 + k)) for k in range(NB)]
    print(f"{'알고리즘':<18}{'평균':>8}{'중앙':>6}{'min':>5}{'max':>5}")
    an = np.array([anneal_worst(b, s*7+2, BUDGET, 'worst') for s, b in boards])
    print(f"{'어닐 (Metropolis)':<18}{an.mean():>8.2f}{int(np.median(an)):>6}{int(an.min()):>5}{int(an.max()):>5}")
    outs = {}
    for L in [20, 50, 100]:
        sc = np.array([lahc(b, s*7+3, BUDGET, L) for s, b in boards])
        outs[L] = sc
        print(f"{'LAHC L=%d'%L:<18}{sc.mean():>8.2f}{int(np.median(sc)):>6}{int(sc.min()):>5}{int(sc.max()):>5}")
    for L in [20, 50, 100]:
        d = outs[L] - an
        print(f"LAHC L={L} vs 어닐: 우세 {int((d>0).sum())} 동점 {int((d==0).sum())} 열세 {int((d<0).sum())} | 평균차 {d.mean():+.2f}")


if __name__ == "__main__":
    main()
