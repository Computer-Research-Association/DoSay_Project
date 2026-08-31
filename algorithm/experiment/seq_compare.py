"""
편집 지점(j) 선택 전략만 격리 비교 — 나머지(rollout·수선·Metropolis·시간예산) 전부 동일.
  first   : j=0 (첫수만; 약체 대조군)
  worst   : j=_pick_worst (production anneal_once 방식)
  uniform : j 균일 랜덤 (내가 이긴다고 주장한 방식)
=> uniform이 worst(=production)를 이겨야만 진짜 배포 이득.
사용: python seq_compare.py [budget초] [N] [보드수]
"""
import sys, time, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.board import make_board, apply_move
from models.board_numba import valid_actions_numba as VA

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 20


def _cells(g, m):
    r1, c1, r2, c2 = m
    return int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1]))


def _fewest(g, a):
    return min(a, key=lambda m: _cells(g, m))


def _rollout_tail(g, rng, gp):
    cleared, seq = 0, []
    while True:
        a = VA(g)
        if not a: return cleared, seq
        mv = _fewest(g, a) if rng.random() < gp else a[rng.randrange(len(a))]
        cleared += apply_move(g, mv); seq.append(mv)


def _pick_worst(pcum, rng):
    n = len(pcum) - 1
    w = [(pcum[i + 1] - pcum[i] - 2) ** 2 + 0.1 for i in range(n)]
    r = rng.random() * sum(w); acc = 0.0
    for i, wi in enumerate(w):
        acc += wi
        if acc >= r: return i
    return n - 1


def anneal(board, seed, budget, point, T0=3.0, gp=0.8):
    rng = random.Random(seed)
    acts = VA(board)
    if not acts: return 0
    g = board.copy(); c0 = apply_move(g, _fewest(board, acts))
    tc, ts = _rollout_tail(g, rng, gp)
    cur_s, cur_S = c0 + tc, [_fewest(board, acts)] + ts
    best = cur_s
    t0 = time.time()
    while time.time() - t0 < budget:
        L = len(cur_S)
        if L < 2: break
        T = max(1e-6, T0 * (1 - (time.time() - t0) / budget))
        if point == 'first':
            j = 0
        elif point == 'uniform':
            j = rng.randrange(L)
        else:  # worst
            gg = board.copy(); pcum = [0]
            for mv in cur_S: pcum.append(pcum[-1] + apply_move(gg, mv))
            j = _pick_worst(pcum, rng)
        g = board.copy(); cleared = 0
        for t in range(j):
            cleared += apply_move(g, cur_S[t])
        a = VA(g)
        if not a: continue
        forbidden = cur_S[j]
        choices = [m for m in a if m != forbidden] or a
        mv = choices[rng.randrange(len(choices))]
        cleared += apply_move(g, mv)
        tc, ts = _rollout_tail(g, rng, gp)
        new_s, new_S = cleared + tc, cur_S[:j] + [mv] + ts
        d = new_s - cur_s
        if d >= 0 or rng.random() < np.exp(d / T):
            cur_s, cur_S = new_s, new_S
        if new_s > best: best = new_s
    return best


def run(point):
    return np.array([max(anneal(make_board(400000 + k), (400000 + k) * 100 + i, BUDGET, point)
                         for i in range(N)) for k in range(NB)])


def main():
    VA(make_board(1))
    print(f"=== 9×18 {NB}판 · 편집지점 전략 격리비교 (budget={BUDGET}s, N={N}) ===")
    print(f"{'전략':<28}{'평균':>8}{'중앙':>6}{'min':>5}{'max':>5}")
    out = {}
    for pt, tag in [('first', 'first (j=0, 약체대조)'),
                    ('worst', 'worst (_pick_worst=현행)'),
                    ('uniform', 'uniform (j 균일랜덤)')]:
        sc = run(pt); out[pt] = sc
        print(f"{tag:<28}{sc.mean():>8.2f}{int(np.median(sc)):>6}{int(sc.min()):>5}{int(sc.max()):>5}")
    d = out['uniform'] - out['worst']
    print(f"\nuniform vs worst(현행): 우세 {int((d>0).sum())}판 동점 {int((d==0).sum())} "
          f"열세 {int((d<0).sum())} | 평균차 {d.mean():+.2f}")
    print("=> " + ("uniform이 현행 이김 → 배포 이득 O" if d.mean() > 0.3
                   else "uniform ≈ 현행 → 앞선 +5는 strawman이었음"))


if __name__ == "__main__":
    main()
