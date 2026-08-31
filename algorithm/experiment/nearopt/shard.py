"""근최적성 곡선 확장: 여러 크기에서 완전탐색 최적 vs 강한 어닐. 재개 가능."""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import exact_small as ex

shard = int(sys.argv[1]); nsh = int(sys.argv[2])
SIZES = [(5,9),(6,9),(7,9)]   # 45,54,63 셀
NB = 12; BASE = 740000
path = f"/Users/ball103/DoSay/algorithm/experiment/nearopt/shard_{shard}.txt"
done = set()
if os.path.exists(path):
    for l in open(path):
        p = l.split()
        if len(p) >= 2: done.add((p[0], p[1]))
out = open(path, "a")
ji = 0
for (R,C) in SIZES:
    for k in range(NB):
        key = (f"{R}x{C}", str(BASE+k))
        if ji % nsh == shard and key not in done:
            seed = BASE + k
            b = ex.make_small(R, C, seed)
            opt, nodes, to = ex.solve_exact(b, time_budget=150)
            # 강한 어닐: max-of-10 × 2000 (셀당 딜로이보다 많은 연산)
            an = max(ex.anneal_once(b, iters=2000, seed=seed*10+i) for i in range(10))
            out.write(f"{R}x{C} {seed} {R*C} {opt} {an} {1 if to else 0} {nodes}\n"); out.flush()
        ji += 1
out.close()
