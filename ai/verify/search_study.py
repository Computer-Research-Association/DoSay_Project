"""**평가의 질 vs 탐색의 양** — 무엇이 지렛대인지 가리는 측정.

    python ai/verify/search_study.py --rollout     # 오차 0 인 1수 가치의 천장
    python ai/verify/search_study.py --fullbeam    # 에피소드 전체 빔의 폭 효과
    python ai/verify/search_study.py --bestof      # 같은 판 재시도 (어닐링의 재시도)

**왜 이 파일이 있는가.** V6~V12 는 전부 "평가함수를 개선한다" 는 한 줄기였고,
V12b 가 `leftover_mae 0.808` 이라는 매우 정확한 가치를 갖고도 V1.0(조잡한 Φ
휴리스틱)과 **통계적으로 같은 점수**를 냈다. 그 줄기가 끝났다는 뜻이다.

그렇다면 남은 축은 **탐색**인데, 탐색에도 종류가 여럿이다. 이 파일은 그것들을
같은 자로 재서 어디에 투자할지 정하기 위한 것이다.

    깊이       — 몇 수 앞까지 보는가 (V7~V9dB 의 빔서치)
    폭         — 몇 개의 부분 게임을 동시에 끌고 가는가 (어닐링에 가까움)
    평가의 정확도 — 잎을 어떻게 매기는가 (휴리스틱 vs 몬테카를로 롤아웃)

**주의**: 판 수가 적으면(10~20판) 앞쪽 시드가 쉬워서 절대값이 100판 기준보다
2점쯤 높게 나온다. **같은 판 수끼리만 비교할 것.**
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import (BASE_SEED, R, C, apple_counts, c_lookahead, c_min_area,
                       c_hi, c_lo, legal_after, legal_mask, make_grid, play,
                       r_hi, r_lo)


# ── 1. 몬테카를로 롤아웃 = 오차 0 인 1수 가치 ────────────────────────────
def rollout_from(grid, rng, base_policy):
    """base_policy 로 끝까지 두고 얻은 점수. 편향 없는 V^base(s) 표본 하나."""
    board = grid.copy()
    score = 0
    while True:
        idx = np.flatnonzero(legal_mask(board[None])[0])
        if idx.size == 0:
            return score
        pick = base_policy(board, idx, rng)
        score += int(apple_counts(board[None])[0, pick])
        board[r_lo[pick]:r_hi[pick], c_lo[pick]:c_hi[pick]] = 0


def play_rollout(seed, rng, base_policy, top_k, n_roll):
    """후보를 top_k 개로 추린 뒤 각각 n_roll 번 끝까지 두어 보고 평균 최고를 고른다.

    이것이 **정책반복의 한 걸음을 정확히 밟은 것**이다 (Tesauro 의 rollout policy).
    학습된 가치와 달리 편향이 없다 — 그 대신 잡음이 크다(판 안 std ~5).
    """
    grid = make_grid(seed)
    score = moves = 0
    while True:
        idx = np.flatnonzero(legal_mask(grid[None])[0])
        if idx.size == 0:
            break
        candidates = idx[np.argsort(-legal_after(grid, idx))[:top_k]]

        best, best_value = candidates[0], -1.0
        for action in candidates:
            child = grid.copy()
            child[r_lo[action]:r_hi[action], c_lo[action]:c_hi[action]] = 0
            gained = int(apple_counts(grid[None])[0, action])
            value = np.mean([gained + rollout_from(child, rng, base_policy)
                             for _ in range(n_roll)])
            if value > best_value:
                best_value, best = value, action

        score += int(apple_counts(grid[None])[0, best])
        grid[r_lo[best]:r_hi[best], c_lo[best]:c_hi[best]] = 0
        moves += 1
    return score, moves


# ── 2. 에피소드 전체 빔 ──────────────────────────────────────────────────
def play_full_beam(seed, width, w_legal, dedup=True):
    """W개의 부분 게임을 **끝까지 동시에** 끌고 가서 마지막에 최고를 취한다.

    깊이 제한 빔(V7~V9dB)과 두 가지가 다르다.
      1) 모든 빔이 같은 수순 번호에 있으므로 '먹은 사과 + 남은 가치' 비교가 공정하다
         (깊이 제한 빔은 깊이가 다른 잎을 비교해야 한다).
      2) 같은 판에 도달한 서로 다른 수순을 중복 제거할 수 있다 — 지운 칸 집합이
         같으면 점수도 같으므로 폭을 낭비하지 않는다.
    """
    grids = make_grid(seed)[None].copy()
    apples = np.zeros(1)
    best = 0
    while True:
        masks = legal_mask(grids)
        alive = masks.any(axis=1)
        if (~alive).any():
            best = max(best, int(apples[~alive].max()))
        if not alive.any():
            break
        grids, apples, masks = grids[alive], apples[alive], masks[alive]

        node_idx, act_idx = np.nonzero(masks)
        children = grids[node_idx].copy()
        removed = apple_counts(grids)[node_idx, act_idx]
        for k in range(node_idx.size):
            a = act_idx[k]
            children[k, r_lo[a]:r_hi[a], c_lo[a]:c_hi[a]] = 0
        child_apples = apples[node_idx] + removed

        if dedup:
            _, keep = np.unique(children.reshape(len(children), -1),
                                axis=0, return_index=True)
            children, child_apples = children[keep], child_apples[keep]

        score = child_apples + w_legal * legal_mask(children).sum(axis=1)
        order = np.argsort(-score)[:width]
        grids, apples = children[order], child_apples[order]
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--fullbeam", action="store_true")
    parser.add_argument("--bestof", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    if args.rollout:
        print("=== 몬테카를로 롤아웃 = 오차 0 인 1수 가치 ===")
        print("(base 정책을 한 걸음 개선한 결과. 학습 가치와 달리 편향이 없다)")
        for label, base, top_k, n_roll, episodes in (
                ("base=min-area(~106) top8 K=1", c_min_area, 8, 1, args.episodes),
                ("base=min-area       top8 K=3", c_min_area, 8, 3, max(args.episodes // 2, 6)),
                ("base=1수앞(~115)    top6 K=1", c_lookahead, 6, 1, max(args.episodes // 2, 6))):
            start = time.time()
            out = [play_rollout(BASE_SEED + i, np.random.default_rng(31 + i),
                                base, top_k, n_roll) for i in range(episodes)]
            scores = np.array([o[0] for o in out], float)
            print(f"  {label:<30}{scores.mean():>7.2f}  "
                  f"({episodes}판, {np.mean([o[1] for o in out]):.1f}수, "
                  f"{time.time() - start:.0f}s)", flush=True)

    if args.fullbeam:
        print("\n=== 에피소드 전체 빔 (평가 고정: 사과 + 2.5 x 합법수) ===")
        print(f"{'폭':>8}{'점수':>9}{'시간':>9}")
        for width in (1, 8, 32, 128, 512):
            start = time.time()
            scores = [play_full_beam(BASE_SEED + i, width, 2.5)
                      for i in range(args.episodes)]
            print(f"{width:>8}{np.mean(scores):>9.2f}{time.time() - start:>8.0f}s", flush=True)

    if args.bestof:
        print("\n=== best-of-N : 같은 판 재시도 (어닐링의 재시도에 해당) ===")
        for label, policy, episodes, widths in (
                ("min-area", c_min_area, args.episodes, (1, 8, 32, 128)),
                ("1수앞", c_lookahead, max(args.episodes // 2, 6), (1, 8, 32))):
            for n in widths:
                start = time.time()
                best = [max(play(BASE_SEED + i, np.random.default_rng(9000 + 97 * i + k),
                                 policy)[0] for k in range(n))
                        for i in range(episodes)]
                print(f"  {label:<10} best-of-{n:<4}{np.mean(best):>8.2f}   "
                      f"({episodes}판, {time.time() - start:.0f}s)", flush=True)

    if not (args.rollout or args.fullbeam or args.bestof):
        parser.error("--rollout / --fullbeam / --bestof 중 하나를 고르세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
