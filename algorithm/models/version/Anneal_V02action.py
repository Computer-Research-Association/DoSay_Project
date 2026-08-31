# 정책: 남은 유효액션 개수 최대화 (단일 피처) — 벡터화 고속판
#  - 각 수를 둔 뒤 valid_actions 수가 가장 많이 남는 수 선택 (선택지 넓게 유지)
#  - 판의 사각형(~7700개)을 numpy로 일괄 평가 → Python 4중포문 대비 수십~수백배 빠름
import numpy as np

GRID = (9, 18)
H, W = GRID

# 모든 (r1<=r2, c1<=c2) 조합 인덱스 (모듈 로드 시 1회 계산)
_r1, _r2, _c1, _c2 = [], [], [], []
for r1 in range(H):
    for r2 in range(r1, H):
        for c1 in range(W):
            for c2 in range(c1, W):
                _r1.append(r1); _r2.append(r2); _c1.append(c1); _c2.append(c2)
R1 = np.array(_r1); R2 = np.array(_r2); C1 = np.array(_c1); C2 = np.array(_c2)


def _count_valid(grid):
    """합=10인 '최소 사각형' 유효 수 개수 (벡터화)."""
    P = np.zeros((H + 1, W + 1), dtype=np.int32)
    P[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)

    def rect(a, b, c, d):   # 합(a..b행, c..d열)
        return P[b + 1, d + 1] - P[a, d + 1] - P[b + 1, c] + P[a, c]

    s = rect(R1, R2, C1, C2)                       # 전체 합
    top = rect(R1, R1, C1, C2)                     # 4변 strip 합
    bot = rect(R2, R2, C1, C2)
    left = rect(R1, R2, C1, C1)
    right = rect(R1, R2, C2, C2)
    valid = (s == 10) & (top > 0) & (bot > 0) & (left > 0) & (right > 0)
    return int(valid.sum())


def rollout_policy(grid, actions, rng):
    best, best_n = None, -1
    for a in actions:
        r1, c1, r2, c2 = a
        region = grid[r1:r2 + 1, c1:c2 + 1]
        saved = region.copy()
        region[:] = 0
        n = _count_valid(grid)          # 이 수를 둔 뒤 남는 유효액션 수
        region[:] = saved               # 원복
        if n > best_n:
            best_n, best = n, a
    return best
