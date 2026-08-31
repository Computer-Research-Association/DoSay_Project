"""
Rollout 기반 beam search: 부분수순을 '휴리스틱' 아닌 '실제 rollout 점수'로 평가해 상위 W 유지.
각 후보는 끝까지 플레이한 실제 획득 점수로 랭킹 → best = 어디서든 본 최고 rollout 점수.
어닐과 동일 벽시계 시간 비교.
사용: python beam_rollout.py [budget초] [보드수]
"""
import sys, time, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.board import make_board, apply_move
from models.board_numba import valid_actions_numba as VA
from seq_compare import anneal as anneal_worst, _cells, _fewest


def rollout_val(g, rng, R, gp=0.8):
    best = 0
    for _ in range(R):
        gg = g.copy(); c = 0
        while True:
            a = VA(gg)
            if not a: break
            mv = _fewest(gg, a) if rng.random() < gp else a[rng.randrange(len(a))]
            c += apply_move(gg, mv)
        if c > best: best = c
    return best


def beam_rollout(board, seed, budget, W=4, C=6, R=1):
    rng = random.Random(seed)
    best = 0; t0 = time.time()
    while time.time() - t0 < budget:
        beam = [(board.copy(), 0)]
        while beam and time.time() - t0 < budget:
            children = []
            for g, cl in beam:
                a = VA(g)
                if not a:
                    if cl > best: best = cl
                    continue
                cand = sorted(a, key=lambda m: _cells(g, m))[:C]
                for m in cand:
                    ng = g.copy(); c = apply_move(ng, m)
                    val = cl + c + rollout_val(ng, rng, R)
                    if val > best: best = val
                    children.append((ng, cl + c, val))
            if not children: break
            children.sort(key=lambda x: -x[2])
            beam = [(x[0], x[1]) for x in children[:W]]
    return best


def main():
    VA(make_board(1))
    print(f"=== 9×18 {NB}판 · rollout-beam vs 어닐 동일시간 (budget={BUDGET}s) ===")
    boards = [(400000 + k, make_board(400000 + k)) for k in range(NB)]
    an = np.array([anneal_worst(b, s*7+2, BUDGET, 'worst') for s, b in boards])
    print(f"{'알고리즘':<22}{'평균':>8}{'중앙':>6}{'min':>5}{'max':>5}")
    print(f"{'어닐 (SA)':<22}{an.mean():>8.2f}{int(np.median(an)):>6}{int(an.min()):>5}{int(an.max()):>5}")
    outs = {}
    for W, C, R in [(4, 6, 1), (8, 6, 1), (4, 8, 2)]:
        sc = np.array([beam_rollout(b, s*7+3, BUDGET, W, C, R) for s, b in boards])
        outs[(W, C, R)] = sc
        print(f"{'beam W%d C%d R%d'%(W,C,R):<22}{sc.mean():>8.2f}{int(np.median(sc)):>6}{int(sc.min()):>5}{int(sc.max()):>5}")
    for cfg in outs:
        d = outs[cfg] - an
        print(f"beam {cfg} vs 어닐: 우세 {int((d>0).sum())} 동점 {int((d==0).sum())} 열세 {int((d<0).sum())} | 평균차 {d.mean():+.2f}")


BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 12
if __name__ == "__main__":
    main()
