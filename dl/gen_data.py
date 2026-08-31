"""
DL 정책 파일럿 — 학습 데이터 생성.

타깃 = 각 (보드상태, 후보수)에 대해 "그 수를 둔 뒤 남는 유효액션 수" (= action-max 기준).
빠른 신경망이 이걸 예측하면 느린 action-max 정책을 밀리초에 재현.

- 상태 다양화: fewest-cells 정책 + 랜덤 섞어 게임을 플레이하며 도중 상태 수집.
- 각 상태의 모든 후보 수에 대해 타깃 계산 (벡터화 카운트라 빠름).
- 저장: boards[N,9,18], rect[N,4], target[N]  → dl/data.npz

사용법: python dl/gen_data.py --games 80
"""
import os, sys, argparse, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

from algorithm.experiment.anneal import _make_grid, _valid_actions, _apply   # noqa: E402
from algorithm.models.version.Anneal_V02action import _count_valid            # noqa: E402  (벡터화)


def gen(games, base_seed=5000, sample_actions=12):
    boards, rects, targets = [], [], []
    rng = random.Random(0)
    for gi in range(games):
        g = _make_grid(base_seed + gi)
        while True:
            acts = _valid_actions(g)
            if not acts:
                break
            # 이 상태의 후보들 (너무 많으면 샘플) → 각 타깃 계산
            chosen = acts if len(acts) <= sample_actions else rng.sample(acts, sample_actions)
            snap = g.copy()
            for a in chosen:
                r1, c1, r2, c2 = a
                region = g[r1:r2+1, c1:c2+1]; saved = region.copy(); region[:] = 0
                t = _count_valid(g)
                region[:] = saved
                boards.append(snap.copy()); rects.append((r1, c1, r2, c2)); targets.append(t)
            # 다음 상태로: 대부분 fewest-cells, 가끔 랜덤 (상태 다양화)
            if rng.random() < 0.3:
                a = acts[rng.randrange(len(acts))]
            else:
                a = min(acts, key=lambda x: int(np.count_nonzero(g[x[0]:x[2]+1, x[1]:x[3]+1])))
            _apply(g, a)
    return (np.array(boards, dtype=np.int8),
            np.array(rects, dtype=np.int16),
            np.array(targets, dtype=np.int16))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=80)
    args = p.parse_args()
    import time; t0 = time.time()
    boards, rects, targets = gen(args.games)
    np.savez(ROOT / "dl" / "data.npz", boards=boards, rects=rects, targets=targets)
    print(f"샘플 {len(targets):,}개  ({args.games}판, {time.time()-t0:.0f}초)")
    print(f" target 분포: min {targets.min()} max {targets.max()} mean {targets.mean():.1f}")
    print(f" → dl/data.npz 저장")


if __name__ == "__main__":
    main()
