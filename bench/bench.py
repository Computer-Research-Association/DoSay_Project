import sys, time, random, ctypes
sys.path.insert(0, '/Users/ball103/DoSay')
sys.path.insert(0, '/Users/ball103/DoSay/bench')
import numpy as np
from models.board import make_board, valid_actions as va_py, valid_actions_fast as va_np, apply_move
from models.board_numba import valid_actions_numba as va_nb
from board_cy import valid_actions_cy as va_cy

# C via ctypes
lib = ctypes.CDLL('/Users/ball103/DoSay/bench/board_c.so')
lib.valid_c.restype = ctypes.c_int
lib.valid_c.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_out = (ctypes.c_int * (512 * 4))()
def va_c(grid):
    g = np.ascontiguousarray(grid, dtype=np.int8)
    n = lib.valid_c(g.ctypes.data_as(ctypes.c_void_p), _out)
    return [(_out[i*4], _out[i*4+1], _out[i*4+2], _out[i*4+3]) for i in range(n)]

methods = [('순수파이썬', va_py), ('numpy벡터', va_np), ('numba', va_nb), ('Cython', va_cy), ('순수C', va_c)]

# 워밍업(JIT)
va_nb(make_board(1)); va_cy(make_board(1)); va_c(make_board(1))

# 정확성
print('=== 정확성 (기준=순수파이썬) ===')
ok = True
for seed in range(1234, 1244):
    g = make_board(seed); rng = random.Random(seed)
    for _ in range(6):
        ref = set(va_py(g))
        for name, f in methods[1:]:
            if set(f(g)) != ref:
                ok = False; print(f'  MISMATCH {name} seed{seed}')
        a = va_py(g)
        if not a: break
        apply_move(g, a[rng.randrange(len(a))])
print('  전부 일치 ✅' if ok else '  불일치 ❌')

# 속도 (밀도별)
print('\n=== 속도 (N=5000회, vs 순수파이썬) ===')
for target, label in [(0.0, '꽉참(162칸)'), (0.5, '중간(80칸)')]:
    gg = make_board(1236); rng = random.Random(0)
    while (gg != 0).sum() > 162 * (1 - target):
        a = va_py(gg)
        if not a: break
        apply_move(gg, a[rng.randrange(len(a))])
    print(f'-- {label} --')
    base = None
    for name, f in methods:
        t = time.time()
        for _ in range(5000): f(gg)
        dt = time.time() - t
        if base is None: base = dt
        print(f'  {name:10s} {dt:6.3f}s   {base/dt:6.1f}배')
