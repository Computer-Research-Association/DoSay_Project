"""동일 시드 1000판에서 SA/ILS/Tabu/rollout-beam 페어드 비교. 재개 가능."""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EXP))
from models.board import make_board
from models.board_numba import valid_actions_numba as VA
from seq_compare import anneal as anneal_worst
from ils import ils
from search_zoo import tabu
from beam_rollout import beam_rollout

shard = int(sys.argv[1]); nsh = int(sys.argv[2])
BUD = 2.5; N = 1000; BASE = 500000
VA(make_board(1))
path = str(EXP / "cmp1000" / f"shard_{shard}.txt")
done = set()
if os.path.exists(path):
    for l in open(path):
        p = l.split()
        if len(p) == 5:
            try: done.add(int(p[0]))
            except: pass
out = open(path, "a")
for i in range(N):
    if i % nsh != shard: continue
    seed = BASE + i
    if seed in done: continue
    b = make_board(seed)
    sa = anneal_worst(b, seed*10+1, BUD, 'worst')
    il = ils(b, seed*10+2, BUD, 15, 3)
    tb = tabu(b, seed*10+3, BUD)
    bm = beam_rollout(b, seed*10+4, BUD, 4, 6, 1)
    out.write(f"{seed} {sa} {il} {tb} {bm}\n"); out.flush()
out.close()
