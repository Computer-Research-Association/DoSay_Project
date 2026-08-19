"""휴리스틱 기준선과 게임의 통계 구조를 **벤치마크와 같은 시드**로 측정한다.

    python ai/verify/baselines.py                 # 기준선 표
    python ai/verify/baselines.py --sweep         # '선택지 vs 즉시사과' 가중치 쓸기
    python ai/verify/baselines.py --spread        # 형제 수 사이의 폭 (docs §3 의 근거)

**왜 이 파일이 저장소에 있는가.** 문서에 오래 적혀 있던 기준선(무작위 96.2 /
탐욕 91.0 / 작은넓이 108.5 / 1수앞 118.1)이 **재현되지 않았다.** 특히 1수앞은
실제로 113.30 이고, 그 4.8점 차이가 V11c 의 설계 동기였다. 기준선은 모델 평가의
좌표축이므로 **언제든 다시 잴 수 있어야 한다.**

합법수 판정을 numpy 로 벡터화했다. game.Board 는 판정 한 번에 26k 번의 파이썬
반복을 돌기 때문에 1수앞 휴리스틱(후보마다 판정 한 번)을 100판 재는 데만 몇십 분이
걸린다. 여기서는 누적합 + 모서리 뺄셈으로 7533개 행동을 한 번에 판정해 78초면 된다.
**판정이 game.Board 와 일치하는지는 실행할 때마다 대조한다.**
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from game.board import Board

R, C = 9, 18
BASE_SEED = 1234

# 모든 직사각형 (1x1 제외) — ai/envs/action_sets.get_all_action 과 같은 순서
_acts = [(r1, c1, r2, c2)
         for r1 in range(R) for r2 in range(r1, R)
         for c1 in range(C) for c2 in range(c1, C)
         if not (r1 == r2 and c1 == c2)]
r_lo = np.array([a[0] for a in _acts])
c_lo = np.array([a[1] for a in _acts])
r_hi = np.array([a[2] for a in _acts]) + 1
c_hi = np.array([a[3] for a in _acts]) + 1
area = (r_hi - r_lo) * (c_hi - c_lo)


def prefix(grids: np.ndarray) -> np.ndarray:
    """(M, R, C) -> (M, R+1, C+1) 0 패딩 2차원 누적합."""
    p = np.zeros((grids.shape[0], R + 1, C + 1), dtype=np.int32)
    np.cumsum(grids, axis=1, out=p[:, 1:, 1:])
    np.cumsum(p[:, 1:, 1:], axis=2, out=p[:, 1:, 1:])
    return p


def _rect(p, a, b, c, d):
    return p[:, a, c] - p[:, b, c] - p[:, a, d] + p[:, b, d]


def legal_mask(grids: np.ndarray) -> np.ndarray:
    """(M, R, C) -> (M, 7533) bool. game.Board 와 같은 판정 (합 10 + 네 변 비지 않음)."""
    p = prefix(grids)
    total = _rect(p, r_hi, r_lo, c_hi, c_lo)
    top = _rect(p, r_lo + 1, r_lo, c_hi, c_lo)
    bottom = _rect(p, r_hi, r_hi - 1, c_hi, c_lo)
    left = _rect(p, r_hi, r_lo, c_lo + 1, c_lo)
    right = _rect(p, r_hi, r_lo, c_hi, c_hi - 1)
    return (total == 10) & (top > 0) & (bottom > 0) & (left > 0) & (right > 0)


def apple_counts(grids: np.ndarray) -> np.ndarray:
    return _rect(prefix((grids != 0).astype(np.int32)), r_hi, r_lo, c_hi, c_lo)


def legal_after(grid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """각 후보 수를 둔 뒤 남는 합법수 개수. (len(idx),)"""
    boards = np.repeat(grid[None], len(idx), axis=0)
    for k, i in enumerate(idx):
        boards[k, r_lo[i]:r_hi[i], c_lo[i]:c_hi[i]] = 0
    return legal_mask(boards).sum(axis=1)


def _pick(idx, values, rng, want_max=True):
    v = values if want_max else -values
    best = np.flatnonzero(v == v.max())
    return idx[best[rng.integers(len(best))]]


def c_random(grid, idx, rng):      return idx[rng.integers(len(idx))]
def c_min_area(grid, idx, rng):    return _pick(idx, area[idx], rng, False)
def c_max_apple(grid, idx, rng):   return _pick(idx, apple_counts(grid[None])[0, idx], rng, True)
def c_min_apple(grid, idx, rng):   return _pick(idx, apple_counts(grid[None])[0, idx], rng, False)
def c_lookahead(grid, idx, rng):   return _pick(idx, legal_after(grid, idx), rng, True)


def c_look_minapple(grid, idx, rng):
    """1수앞 최대, 동점이면 사과를 적게 먹는 쪽. 1수앞 계열의 최고 조합."""
    n = legal_after(grid, idx)
    cand = idx[n == n.max()]
    return _pick(cand, apple_counts(grid[None])[0, cand], rng, False)


def blend(lam: float):
    """남는 합법수 + lam x 즉시 사과 수. lam 을 키우면 최다 제거 탐욕이 된다."""
    def chooser(grid, idx, rng):
        n = legal_after(grid, idx).astype(float)
        a = apple_counts(grid[None])[0, idx].astype(float)
        return _pick(idx, n + lam * a, rng, True)
    return chooser


def make_grid(board_seed: int) -> np.ndarray:
    """`game.Board.from_seed` 와 **완전히 같은 판**을 만든다.

    주의: `dtype=np.int8` 을 빼먹으면 다른 판이 나온다. numpy 의 Generator 는
    유계 정수를 뽑을 때 dtype 마다 다른 알고리즘(워드 크기)을 쓰기 때문에
    같은 seed 라도 스트림이 갈린다. 이걸 놓치면 "같은 시드로 쟀다" 고 믿으면서
    실제로는 다른 판을 재게 된다 (2026-08-10 에 실제로 그렇게 틀렸다).
    """
    return np.random.default_rng(board_seed).integers(
        1, 10, size=(R, C), dtype=np.int8).astype(np.int32)


def play(board_seed: int, rng, chooser):
    grid = make_grid(board_seed)
    score = moves = 0
    while True:
        idx = np.flatnonzero(legal_mask(grid[None])[0])
        if idx.size == 0:
            break
        pick = chooser(grid, idx, rng)
        score += int(apple_counts(grid[None])[0, pick])
        grid[r_lo[pick]:r_hi[pick], c_lo[pick]:c_hi[pick]] = 0
        moves += 1
    return score, moves, grid


def _run(chooser, episodes):
    scores, moves, residue = [], [], np.zeros(10)
    for i in range(episodes):
        s, m, grid = play(BASE_SEED + i, np.random.default_rng(777 + i), chooser)
        scores.append(s)
        moves.append(m)
        values, counts = np.unique(grid[grid > 0], return_counts=True)
        residue[values] += counts
    return np.array(scores, float), float(np.mean(moves)), residue / episodes



# ══════════════════════════════════════════════════════════════════════════
# 탐색 깊이의 효과 — **평가함수를 고정한 채** 깊이만 늘리면 얼마나 오르는가
#
# 이것이 V13 의 근거다. V6~V12 는 전부 '평가함수 개선' 이었고 1수 탐욕의 천장
# (116.5)을 못 넘었다. 반면 빔 깊이 2 는 더 나쁜 평가함수로도 121.45 를 냈고,
# 알고리즘 팀의 어닐링은 "최소 사과 제거" 라는 사소한 기준 하나로 ~137 을 낸다.
# 즉 지렛대가 평가가 아니라 **탐색**일 수 있다. 여기서 그것을 직접 잰다.
# ══════════════════════════════════════════════════════════════════════════

def _expand(grids, apples, firsts, masks):
    """(F,R,C) 노드들의 **모든 합법 자식**. (자식판, 누적사과, 첫수)"""
    node_idx, act_idx = np.nonzero(masks)
    if node_idx.size == 0:
        return None
    children = grids[node_idx].copy()
    removed = apple_counts(grids)[node_idx, act_idx]
    for k in range(node_idx.size):
        a = act_idx[k]
        children[k, r_lo[a]:r_hi[a], c_lo[a]:c_hi[a]] = 0
    new_apples = apples[node_idx] + removed
    new_firsts = np.where(firsts[node_idx] < 0, act_idx, firsts[node_idx])
    return children, new_apples, new_firsts


def beam_choose(grid, mask, depth, width, w_legal):
    """빔 탐색으로 첫 수를 고른다. 잎 평가 = 경로 사과 + w x (잎의 합법수).

    평가함수는 손으로 쓴 것 그대로다 — 여기서 재려는 것은 '평가의 질' 이 아니라
    **깊이 그 자체의 값어치**이기 때문이다.
    """
    nodes = grid[None].copy()
    apples = np.zeros(1)
    firsts = np.full(1, -1)
    node_masks = mask[None]
    best_score, best_first = -np.inf, -1

    for _ in range(depth):
        out = _expand(nodes, apples, firsts, node_masks)
        if out is None:
            break
        child_grids, child_apples, child_firsts = out
        child_masks = legal_mask(child_grids)
        n_legal = child_masks.sum(axis=1)

        score = child_apples + w_legal * n_legal
        top = int(np.argmax(score))
        if score[top] > best_score:
            best_score, best_first = float(score[top]), int(child_firsts[top])

        alive = np.flatnonzero(n_legal > 0)          # 끝난 판은 더 펼칠 것이 없다
        if alive.size == 0:
            break
        order = alive[np.argsort(-score[alive])][:width]
        nodes, apples, firsts = child_grids[order], child_apples[order], child_firsts[order]
        node_masks = child_masks[order]

    return best_first


def play_beam(board_seed, depth, width, w_legal):
    grid = make_grid(board_seed)
    score = moves = 0
    while True:
        mask = legal_mask(grid[None])[0]
        if not mask.any():
            break
        pick = beam_choose(grid, mask, depth, width, w_legal)
        if pick < 0:
            break
        score += int(apple_counts(grid[None])[0, pick])
        grid[r_lo[pick]:r_hi[pick], c_lo[pick]:c_hi[pick]] = 0
        moves += 1
    return score, moves


def check_engine() -> tuple[bool, bool]:
    """(판 생성이 엔진과 같은가, 합법수 판정이 엔진과 같은가). 매 실행마다 대조한다.

    판 생성 대조가 여기 있는 이유: 2026-08-10 에 `dtype=np.int8` 을 빼먹어
    **다른 판**에서 기준선을 재고도 "같은 시드로 쟀다" 고 믿은 적이 있다.
    합법수 판정만 맞으면 조용히 틀리는 종류의 실수라 자동 대조가 필요하다.
    """
    grids_ok = legal_ok = True
    for s in range(5):
        board = Board.from_seed((R, C), 3000 + s)
        engine = board.grid.astype(np.int32)
        grids_ok &= bool(np.array_equal(engine, make_grid(3000 + s)))
        legal_ok &= len(board.get_valid_actions()) == int(legal_mask(engine[None]).sum())
    return grids_ok, legal_ok


POLICIES = [
    ("최다 제거 탐욕", c_max_apple),
    ("무작위", c_random),
    ("최소 넓이", c_min_area),
    ("최소 제거 (짝짓기 근사)", c_min_apple),
    ("1수앞 합법수 최대", c_lookahead),
    ("1수앞 + 최소제거 타이브레이크", c_look_minapple),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="선택지 vs 즉시사과 가중치 쓸기")
    parser.add_argument("--spread", action="store_true", help="형제 수 사이의 폭")
    parser.add_argument("--beam", action="store_true", help="탐색 깊이의 효과")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3, 4, 6])
    parser.add_argument("--w-legal", type=float, nargs="*", default=[0.3])
    args = parser.parse_args()

    grids_ok, legal_ok = check_engine()
    print(f"엔진 대조 — 판 생성: {'일치' if grids_ok else '불일치!'} / "
          f"합법수 판정: {'일치' if legal_ok else '불일치!'}")
    if not (grids_ok and legal_ok):
        raise SystemExit("엔진과 어긋납니다. 이 상태의 측정값은 벤치마크와 비교할 수 없습니다.")
    print()

    if args.beam:
        print(f"빔 탐색 (평가함수 고정: 경로사과 + w x 잎 합법수), width={args.width}")
        print(f"{'설정':<28}{'점수':>8}{'수':>7}{'점/수':>8}{'std':>7}   시간")
        for w_legal in args.w_legal:
            for depth in args.depths:
                start = time.time()
                out = [play_beam(BASE_SEED + i, depth, args.width, w_legal)
                       for i in range(args.episodes)]
                scores = np.array([o[0] for o in out], float)
                moves = float(np.mean([o[1] for o in out]))
                print(f"  depth={depth}  w={w_legal:<14}{scores.mean():>7.2f}{moves:>7.1f}"
                      f"{scores.mean()/moves:>8.3f}{scores.std():>7.1f}   {time.time()-start:.0f}s",
                      flush=True)
        return 0

    if args.spread:
        # docs/game-analysis.md §3 의 근거. 한 국면 안에서 합법수들이 얼마나 다른가.
        ap_std, n_std, n_legal = [], [], []
        for i in range(30):
            rng = np.random.default_rng(777 + i)
            grid = make_grid(BASE_SEED + i)
            while True:
                idx = np.flatnonzero(legal_mask(grid[None])[0])
                if idx.size == 0:
                    break
                if idx.size > 1:
                    ap_std.append(apple_counts(grid[None])[0, idx].astype(float).std())
                    n_std.append(legal_after(grid, idx).astype(float).std())
                    n_legal.append(idx.size)
                pick = c_min_area(grid, idx, rng)
                grid[r_lo[pick]:r_hi[pick], c_lo[pick]:c_hi[pick]] = 0
        mean_legal = float(np.mean(n_legal))
        print(f"국면 {len(ap_std)}개, 합법수 평균 {mean_legal:.1f}개")
        print("한 국면 안에서 형제 수들 사이의 폭 (보상 단위 = 사과/162):")
        print(f"  즉시 보상 Δr                    std {np.mean(ap_std) / 162:.4f}"
              f"   (사과 {np.mean(ap_std):.2f}개)")
        print(f"  ΔΦ  (Φ = 0.3·log1p(합법수))     std {0.3 * np.mean(n_std) / mean_legal:.4f}"
              f"   (합법수 {np.mean(n_std):.2f}개)")
        print("  참고: 진짜 형제 간 가치 차이 0.0123, 가치망 추정 오차 0.0103 (V7 측정)")
        return 0

    plans = ([(f"lam = {lam:+.2f}", blend(lam))
              for lam in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)]
             if args.sweep else POLICIES)

    print(f"{'정책':<36}{'점수':>8}{'수':>7}{'점/수':>8}{'std':>7}   시간")
    for name, chooser in plans:
        start = time.time()
        scores, moves, residue = _run(chooser, args.episodes)
        mean = float(scores.mean())
        print(f"  {name:<34}{mean:>7.2f}{moves:>7.1f}{mean / moves:>8.3f}"
              f"{scores.std():>7.1f}   {time.time() - start:.0f}s", flush=True)
        if not args.sweep:
            print("      숫자별 잔여율: "
                  + " ".join(f"{d}:{residue[d] / 18:.0%}" for d in range(1, 10)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
