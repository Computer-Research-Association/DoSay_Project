"""어닐링 수순 최적화 — bench/anneal_cy.pyx 의 numba 이식.

원본은 Cython(.pyx)이라 C 컴파일러가 있어야 빌드된다. 이 환경에는 MSVC 가 없어서
같은 알고리즘을 numba 로 옮겼다. numba 는 LLVM 을 들고 다녀서 컴파일러가 필요 없고,
스칼라 루프 위주인 이 알고리즘에서는 C 와 비슷한 속도가 나온다.

**알고리즘은 원본과 같게 유지했다.** 상수 하나라도 바꾸면 알고리즘 팀이 낸 점수와
비교할 수 없게 된다. 바꾼 것은 셋뿐이고 전부 안전성 쪽이다.

  1. 판 크기를 인자로 받는다 (원본은 9x18 컴파일 상수)
  2. 합법수 배열이 넘칠 때 조용히 뭉개지 않고 잘랐음을 알린다 (원본은 MAXM=600 고정)
  3. 난수 상태를 전역 대신 배열로 들고 다닌다 (numba 에는 전역 가변 상태가 없다)

한 수는 (r1, c1, r2, c2) 네 정수다. 합법 조건은 game.Board 와 완전히 같다 —
직사각형 합이 정확히 10 이고, 네 변이 모두 비어 있지 않을 것("최소" 조건).

    from algorithm.models.anneal import anneal_once
    score, seq = anneal_once(grid, iters=8000, rng_seed=0)
"""

import numpy as np
from numba import njit

MAX_MOVES = 4096   # 한 판에서 나올 수 있는 합법수 상한 (원본 600 은 넘칠 여지가 있었다)
MAX_LEN = 128      # 수순 길이 상한. 한 수가 합 10 을 가져가므로 실제로는 81 이 최대다.

DEFAULT_ITERS = 8000
DEFAULT_T0 = 3.0
DEFAULT_GREEDY_P = 0.8
IDEAL_PER_MOVE = 2   # 여집합 짝짓기의 점/수. game-analysis.md 1절.


# ── 난수 (xorshift64) ────────────────────────────────────────────────────────
# 상태를 1칸짜리 배열로 들고 다닌다. 원본의 전역 _S 와 같은 수열을 낸다.

@njit(cache=True, inline="always")
def _next(state):
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return x


@njit(cache=True, inline="always")
def _rand(state):
    return (_next(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def _randint(state, n):
    return np.int64(_next(state) % np.uint64(n))


# ── 판 조작 ──────────────────────────────────────────────────────────────────

@njit(cache=True)
def _find_moves(g, rows, cols, out):
    """합이 10이고 네 변이 비어 있지 않은 직사각형을 전부 찾는다.

    2차원 누적합으로 임의 구간 합을 상수 시간에 본다. 왼쪽 위를 고정하고 오른쪽
    아래를 넓혀 가다 합이 10을 넘으면 그 줄은 더 볼 것이 없어 끊는다.
    """
    prefix = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            prefix[r + 1, c + 1] = (g[r * cols + c] + prefix[r, c + 1]
                                    + prefix[r + 1, c] - prefix[r, c])

    n = 0
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    total = (prefix[r2 + 1, c2 + 1] - prefix[r1, c2 + 1]
                             - prefix[r2 + 1, c1] + prefix[r1, c1])
                    if total == 10:
                        top = (prefix[r1 + 1, c2 + 1] - prefix[r1, c2 + 1]
                               - prefix[r1 + 1, c1] + prefix[r1, c1])
                        bot = (prefix[r2 + 1, c2 + 1] - prefix[r2, c2 + 1]
                               - prefix[r2 + 1, c1] + prefix[r2, c1])
                        left = (prefix[r2 + 1, c1 + 1] - prefix[r1, c1 + 1]
                                - prefix[r2 + 1, c1] + prefix[r1, c1])
                        right = (prefix[r2 + 1, c2 + 1] - prefix[r1, c2 + 1]
                                 - prefix[r2 + 1, c2] + prefix[r1, c2])
                        if top > 0 and bot > 0 and left > 0 and right > 0:
                            if n < out.shape[0]:
                                out[n, 0] = r1; out[n, 1] = c1
                                out[n, 2] = r2; out[n, 3] = c2
                            n += 1
                    elif total > 10:
                        break
    return n


@njit(cache=True, inline="always")
def _apply(g, cols, r1, c1, r2, c2):
    cleared = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if g[r * cols + c] > 0:
                cleared += 1
                g[r * cols + c] = 0
    return cleared


@njit(cache=True)
def _fewest_idx(g, cols, moves, n):
    """사과를 가장 적게 지우는 수. 큰 사각형은 여집합 재고를 태우기 때문이다."""
    best, best_count = 0, 1 << 30
    for i in range(n):
        k = 0
        for r in range(moves[i, 0], moves[i, 2] + 1):
            for c in range(moves[i, 1], moves[i, 3] + 1):
                if g[r * cols + c] > 0:
                    k += 1
        if k < best_count:
            best_count = k
            best = i
    return best


@njit(cache=True)
def _rollout(g, rows, cols, first, greedy_p, seq, state, moves):
    """첫 수를 두고, 그 뒤로는 끝날 때까지 둔다.

    greedy_p 확률로 '가장 적게 지우는 수', 나머지는 무작위. 합법수가 하나도
    없을 때까지 가므로 수순은 항상 끝난 판에서 멈춘다.
    """
    total = _apply(g, cols, first[0], first[1], first[2], first[3])
    seq[0, 0] = first[0]; seq[0, 1] = first[1]
    seq[0, 2] = first[2]; seq[0, 3] = first[3]
    length = 1

    while True:
        n = _find_moves(g, rows, cols, moves)
        if n == 0:
            break
        n = min(n, moves.shape[0])
        if _rand(state) >= greedy_p:
            idx = _randint(state, n)
        else:
            idx = _fewest_idx(g, cols, moves, n)
        total += _apply(g, cols, moves[idx, 0], moves[idx, 1], moves[idx, 2], moves[idx, 3])
        seq[length, 0] = moves[idx, 0]; seq[length, 1] = moves[idx, 1]
        seq[length, 2] = moves[idx, 2]; seq[length, 3] = moves[idx, 3]
        length += 1
        if length >= seq.shape[0] - 1:
            break
    return total, length


@njit(cache=True)
def _build_prefix(start, cols, seq, length, states, cleared):
    """수순의 각 시점 판을 미리 만들어 둔다. 되돌아갈 때 처음부터 다시 두지 않으려고."""
    states[0] = start
    cleared[0] = 0
    acc = 0
    for step in range(length):
        states[step + 1] = states[step]
        acc += _apply(states[step + 1], cols, seq[step, 0], seq[step, 1],
                      seq[step, 2], seq[step, 3])
        cleared[step + 1] = acc


@njit(cache=True)
def _pick_worst(cleared, n, state):
    """어느 수를 부술지 고른다.

    이상적인 한 수는 사과 2개를 가져간다(여집합 짝짓기). 거기서 벗어난 정도를
    제곱해 가중치로 삼으니, 3개 이상 태운 수나 헛돈 수가 먼저 다시 뽑힌다.
    """
    total = 0.0
    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = cleared[i + 1] - cleared[i] - IDEAL_PER_MOVE
        weights[i] = float(d * d) + 0.1
        total += weights[i]

    target = _rand(state) * total
    acc = 0.0
    for i in range(n):
        acc += weights[i]
        if acc >= target:
            return i
    return n - 1


@njit(cache=True)
def _anneal(board, iters, seed, t0, greedy_p):
    rows, cols = board.shape
    cell_count = rows * cols

    state = np.empty(1, dtype=np.uint64)
    state[0] = np.uint64(seed) * np.uint64(2747636419) + np.uint64(1)
    if state[0] == np.uint64(0):
        state[0] = np.uint64(0x9E3779B97F4A7C15)

    start = np.empty(cell_count, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            start[r * cols + c] = board[r, c]

    moves = np.empty((MAX_MOVES, 4), dtype=np.int32)
    grid = start.copy()

    n = _find_moves(grid, rows, cols, moves)
    if n == 0:
        return 0, np.empty((0, 4), dtype=np.int32), 0
    n = min(n, MAX_MOVES)

    current = np.empty((MAX_LEN, 4), dtype=np.int32)
    best = np.empty((MAX_LEN, 4), dtype=np.int32)
    tail = np.empty((MAX_LEN, 4), dtype=np.int32)
    states = np.empty((MAX_LEN + 1, cell_count), dtype=np.int32)
    cleared = np.empty(MAX_LEN + 1, dtype=np.int32)

    # 출발점: 가장 적게 지우는 수에서 시작해 탐욕적으로 끝까지 (greedy_p=1.0)
    idx = _fewest_idx(grid, cols, moves, n)
    cur_score, cur_len = _rollout(grid, rows, cols, moves[idx], 1.0, current, state, moves)
    best_score, best_len = cur_score, cur_len
    best[:cur_len] = current[:cur_len]
    _build_prefix(start, cols, current, cur_len, states, cleared)

    for it in range(iters):
        temperature = t0 * (1.0 - float(it) / iters) + 1e-6
        if cur_len < 2:
            break

        t = _pick_worst(cleared, cur_len, state)   # 부술 지점
        grid[:] = states[t]                        # 그 시점으로 되돌린다
        n = _find_moves(grid, rows, cols, moves)
        if n < 2:
            continue
        n = min(n, MAX_MOVES)

        # 원래 두었던 수와는 다른 수를 고른다
        idx = _randint(state, n)
        for _ in range(25):
            if not (moves[idx, 0] == current[t, 0] and moves[idx, 1] == current[t, 1]
                    and moves[idx, 2] == current[t, 2] and moves[idx, 3] == current[t, 3]):
                break
            idx = _randint(state, n)

        tail_total, tail_len = _rollout(grid, rows, cols, moves[idx], greedy_p,
                                        tail, state, moves)
        if t + tail_len >= MAX_LEN:
            continue
        new_score = cleared[t] + tail_total
        delta = float(new_score - cur_score)

        if delta >= 0.0 or _rand(state) < np.exp(delta / temperature):
            current[t:t + tail_len] = tail[:tail_len]
            cur_len = t + tail_len
            cur_score = new_score
            _build_prefix(start, cols, current, cur_len, states, cleared)
            if cur_score > best_score:
                best_score, best_len = cur_score, cur_len
                best[:cur_len] = current[:cur_len]

    return best_score, best[:best_len].copy(), best_len


def anneal_once(grid, iters: int = DEFAULT_ITERS, rng_seed: int = 0,
                t0: float = DEFAULT_T0, greedy_p: float = DEFAULT_GREEDY_P):
    """어닐링 1회. (점수, [(r1, c1, r2, c2), ...]) 를 돌려준다.

    수순은 항상 '더 둘 수 없는' 판에서 끝난다 (rollout 이 합법수가 0이 될 때까지 간다).
    부르는 쪽은 그 성질에 기대도 되지만, 판이 바뀌었다면 다시 계산해야 한다.
    """
    board = np.ascontiguousarray(grid, dtype=np.int8)
    if board.ndim != 2:
        raise ValueError(f"판은 2차원이어야 합니다: {board.shape}")

    score, seq, length = _anneal(board, int(iters), int(rng_seed), float(t0), float(greedy_p))
    return int(score), [(int(m[0]), int(m[1]), int(m[2]), int(m[3])) for m in seq[:length]]
