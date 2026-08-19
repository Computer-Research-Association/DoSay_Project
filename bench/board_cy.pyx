# cython: boundscheck=False, wraparound=False, cdivision=True
def valid_actions_cy(signed char[:, :] grid):
    cdef int P[10][19]
    cdef int r, c, r1, r2, c1, c2, s, top, bot, left, right
    for r in range(10):
        for c in range(19):
            P[r][c] = 0
    for r in range(9):
        for c in range(18):
            P[r+1][c+1] = grid[r, c] + P[r][c+1] + P[r+1][c] - P[r][c]
    res = []
    for r1 in range(9):
        for r2 in range(r1, 9):
            for c1 in range(18):
                for c2 in range(c1, 18):
                    s = P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]
                    if s == 10:
                        top = P[r1+1][c2+1] - P[r1][c2+1] - P[r1+1][c1] + P[r1][c1]
                        bot = P[r2+1][c2+1] - P[r2][c2+1] - P[r2+1][c1] + P[r2][c1]
                        left = P[r2+1][c1+1] - P[r1][c1+1] - P[r2+1][c1] + P[r1][c1]
                        right = P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c2] + P[r1][c2]
                        if top > 0 and bot > 0 and left > 0 and right > 0:
                            res.append((r1, c1, r2, c2))
                    elif s > 10:
                        break
    return res
