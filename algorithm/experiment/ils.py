"""
ILS(Iterated Local Search) vs 어닐 — 이웃연산자 동일(worst-point), 탐색골격만 다름.
ILS: 국소최적까지 하강(개선만 수용) → kick(강제 k회 편집으로 분지 탈출) → 재하강. best 추적.
사용: python ils.py [budget초] [보드수]
"""
import sys, time, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.board import make_board, apply_move
from models.board_numba import valid_actions_numba as VA
from seq_compare import anneal as anneal_worst, _fewest, _rollout_tail
from lahc import neighbor


def local_search(board, S, s, rng, patience):
    noimp = 0
    while noimp < patience:
        nb = neighbor(board, S, rng)
        if nb is None: break
        ns, nS = nb
        if ns > s:
            s, S = ns, nS; noimp = 0
        else:
            noimp += 1
    return s, S


def kick(board, S, rng, k):
    s = None
    for _ in range(k):
        nb = neighbor(board, S, rng)
        if nb is None: break
        s, S = nb
    if s is None:
        g = board.copy(); s = sum(apply_move(g, m) for m in S)
    return s, S


def ils(board, seed, budget, patience=15, kick_k=3):
    rng = random.Random(seed)
    acts = VA(board)
    if not acts: return 0
    g = board.copy(); c0 = apply_move(g, _fewest(board, acts))
    tc, ts = _rollout_tail(g, rng, 0.8)
    cur_s, cur_S = c0 + tc, [_fewest(board, acts)] + ts
    cur_s, cur_S = local_search(board, cur_S, cur_s, rng, patience)
    best, best_S = cur_s, cur_S
    t0 = time.time()
    while time.time() - t0 < budget:
        ks, kS = kick(board, cur_S, rng, kick_k)
        ls, lS = local_search(board, kS, ks, rng, patience)
        if ls >= cur_s:
            cur_s, cur_S = ls, lS
        elif ls < best - 5:            # 너무 떨어지면 best서 재시작
            cur_s, cur_S = best, best_S
        if ls > best: best, best_S = ls, lS
    return best


def main():
    VA(make_board(1))
    print(f"=== 9×18 {NB}판 · ILS vs 어닐 동일시간 (budget={BUDGET}s) ===")
    boards = [(400000 + k, make_board(400000 + k)) for k in range(NB)]
    print(f"{'알고리즘':<20}{'평균':>8}{'중앙':>6}{'min':>5}{'max':>5}")
    an = np.array([anneal_worst(b, s*7+2, BUDGET, 'worst') for s, b in boards])
    print(f"{'어닐 (SA)':<20}{an.mean():>8.2f}{int(np.median(an)):>6}{int(an.min()):>5}{int(an.max()):>5}")
    outs = {}
    for pat, kk in [(10, 2), (15, 3), (25, 4)]:
        sc = np.array([ils(b, s*7+3, BUDGET, pat, kk) for s, b in boards])
        outs[(pat, kk)] = sc
        print(f"{'ILS pat=%d k=%d'%(pat,kk):<20}{sc.mean():>8.2f}{int(np.median(sc)):>6}{int(sc.min()):>5}{int(sc.max()):>5}")
    for cfg in outs:
        d = outs[cfg] - an
        print(f"ILS {cfg} vs 어닐: 우세 {int((d>0).sum())} 동점 {int((d==0).sum())} 열세 {int((d<0).sum())} | 평균차 {d.mean():+.2f}")


BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 12
if __name__ == "__main__":
    main()
