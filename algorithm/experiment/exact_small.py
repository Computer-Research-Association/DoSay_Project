"""
소형 보드 완전탐색(branch & bound + transposition table)으로 '진짜 최적해'를 구하고,
어닐링이 그 증명된 천장에 얼마나 붙는지 검증한다.

- valid_actions: 임의 크기 보드용 (합=10 최소 사각형)
- solve_exact: DFS + 메모이제이션(잔여보드→최대 추가획득) + 상계 가지치기
              => 반환값은 '증명된 최적 제거칸수'
- anneal_small: 소형 보드용 간이 어닐링(worst-neighbor + max-of-N)
사용: python exact_small.py [R] [C] [N보드] [seed0]
"""
import sys, time, random
import numpy as np

# ---------- 임의 크기 valid_actions ----------
def prefix(g):
    R, C = g.shape
    P = np.zeros((R + 1, C + 1), dtype=np.int64)
    P[1:, 1:] = np.cumsum(np.cumsum(g, axis=0), axis=1)
    return P

def valid_actions(g):
    R, C = g.shape
    P = prefix(g)
    def area(r1, c1, r2, c2):
        return int(P[r2 + 1, c2 + 1] - P[r1, c2 + 1] - P[r2 + 1, c1] + P[r1, c1])
    res = []
    for r1 in range(R):
        for r2 in range(r1, R):
            for c1 in range(C):
                for c2 in range(c1, C):
                    s = area(r1, c1, r2, c2)
                    if s == 10:
                        if area(r1, c1, r1, c2) == 0: continue
                        if area(r1, c2, r2, c2) == 0: continue
                        if area(r2, c1, r2, c2) == 0: continue
                        if area(r1, c1, r2, c1) == 0: continue
                        res.append((r1, c1, r2, c2))
                    elif s > 10:
                        break
    return res

def apply_move(g, mv):
    r1, c1, r2, c2 = mv
    reg = g[r1:r2 + 1, c1:c2 + 1]
    cl = int(np.count_nonzero(reg))
    reg[:] = 0
    return cl


# ---------- 완전탐색 (증명된 최적) ----------
def solve_exact(board, time_budget=None):
    """반환: (최적 제거칸수, 방문상태수, 타임아웃여부)"""
    memo = {}
    stats = {"nodes": 0}
    t0 = time.time()
    timed_out = [False]

    def upper_bound(g):
        # 남은 사과 중 '합=10 사각형에 낄 수 있는' 칸의 상한 = 그냥 남은 사과수(느슨하지만 안전)
        return int(np.count_nonzero(g))

    def dfs(g):
        if time_budget and time.time() - t0 > time_budget:
            timed_out[0] = True
            return 0
        key = g.tobytes()
        if key in memo:
            return memo[key]
        stats["nodes"] += 1
        acts = valid_actions(g)
        if not acts:
            memo[key] = 0
            return 0
        best = 0
        for mv in acts:
            r1, c1, r2, c2 = mv
            saved = g[r1:r2 + 1, c1:c2 + 1].copy()
            cl = apply_move(g, mv)
            # 상계 가지치기: 지금 제거 + 남은 전부 따도 best 못 넘으면 스킵
            if cl + upper_bound(g) > best:
                val = cl + dfs(g)
                if val > best:
                    best = val
            g[r1:r2 + 1, c1:c2 + 1] = saved
            if timed_out[0]:
                break
        memo[key] = best
        return best

    best = dfs(board.copy())
    return best, stats["nodes"], timed_out[0]


# ---------- 소형 어닐링 (worst-neighbor + max-of-N) ----------
def _fewest(g, mv):
    r1, c1, r2, c2 = mv
    return int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1]))

def rollout(g, first, rng, gp=0.8):
    total = apply_move(g, first)
    while True:
        a = valid_actions(g)
        if not a: return total
        mv = min(a, key=lambda m: _fewest(g, m)) if rng.random() < gp else a[rng.randrange(len(a))]
        total += apply_move(g, mv)

def anneal_once(board, iters, seed, T0=3.0, gp=0.8):
    rng = random.Random(seed)
    acts = valid_actions(board)
    if not acts:
        return 0
    # 현재 수순: 첫 수 인덱스만 정하고 rollout
    def score_of(first):
        return rollout(board.copy(), first, rng, gp)
    cur_first = acts[rng.randrange(len(acts))]
    cur = score_of(cur_first)
    best = cur
    for i in range(iters):
        T = T0 * (1 - i / iters) + 1e-6
        cand = acts[rng.randrange(len(acts))]
        s = score_of(cand)
        if s >= cur or rng.random() < np.exp((s - cur) / T):
            cur, cur_first = s, cand
        if s > best:
            best = s
    return best

def anneal_small(board, iters=2000, instances=8, seed=0):
    return max(anneal_once(board, iters, seed * 100 + i) for i in range(instances))


# ---------- 소형 보드 생성 ----------
def make_small(R, C, seed):
    return np.random.default_rng(seed).integers(1, 10, (R, C)).astype(np.int8)


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 700000
    total = R * C
    print(f"=== {R}×{C} 보드 {N}개 : 완전탐색 최적 vs 어닐링 ===")
    print(f"{'seed':>7} {'셀수':>4} {'최적':>4} {'어닐':>4} {'gap':>4} {'노드수':>10} {'탐색초':>7}")
    gaps = []; opt_hit = 0; perfect_opt = 0
    for k in range(N):
        seed = seed0 + k
        b = make_small(R, C, seed)
        t0 = time.time()
        opt, nodes, to = solve_exact(b, time_budget=60)
        te = time.time() - t0
        an = anneal_small(b, iters=1500, instances=8, seed=seed)
        gap = opt - an
        gaps.append(gap)
        if gap == 0: opt_hit += 1
        if opt == total: perfect_opt += 1
        flag = "⏱TIMEOUT" if to else ""
        print(f"{seed:>7} {total:>4} {opt:>4} {an:>4} {gap:>4} {nodes:>10,} {te:>7.2f} {flag}")
    g = np.array(gaps)
    print(f"\n[요약] 어닐=최적 {opt_hit}/{N}판 ({opt_hit/N*100:.0f}%)   "
          f"평균 gap {g.mean():.2f}   최대 gap {int(g.max())}")
    print(f"       완전제거 가능판(최적=셀수): {perfect_opt}/{N}")


if __name__ == "__main__":
    main()
