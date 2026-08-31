import sys, os
sys.path.insert(0,'/Users/ball103/DoSay')
import models.beam as B
from models.board_numba import valid_actions_numba as VA
B.valid_actions=VA
from models.board import make_board
sh=int(sys.argv[1]); nsh=int(sys.argv[2]); N=1000; S0=1234
VA(make_board(1))
configs=[(2,2),(3,3),(3,5),(5,3),(5,5)]
path=f'/Users/ball103/DoSay/algorithm/experiment/measure1000/beam_{sh}.txt'
done=set()
if os.path.exists(path):
    for l in open(path):
        p=l.split()
        if len(p)==3: done.add((int(p[0]),p[1]))
out=open(path,'a')
for i in range(N):
    if i%nsh!=sh: continue
    seed=S0+i
    for w,d in configs:
        key=(seed,f'{w}_{d}')
        if key in done: continue
        sc=B.play(make_board(seed),w,d)
        out.write(f'{seed} {w}_{d} {sc}\n'); out.flush()
out.close()
