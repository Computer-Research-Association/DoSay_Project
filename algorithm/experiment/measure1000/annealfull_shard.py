import sys, os
sys.path.insert(0,'/Users/ball103/DoSay'); sys.path.insert(0,'/Users/ball103/DoSay/bench')
import numpy as np
from models.board import make_board
from anneal_cy import anneal_cy
sh=int(sys.argv[1]); nsh=int(sys.argv[2]); N=1000; S0=1234
path=f'/Users/ball103/DoSay/algorithm/experiment/measure1000/annealfull_{sh}.txt'
done=set()
if os.path.exists(path):
    for l in open(path):
        p=l.split()
        if p: done.add(int(p[0]))
out=open(path,'a')
for i in range(N):
    if i%nsh!=sh: continue
    seed=S0+i
    if seed in done: continue
    b=np.ascontiguousarray(make_board(seed))
    best=max(anneal_cy(b,10000,seed*100+j)[0] for j in range(16))
    out.write(f'{seed} {best}\n'); out.flush()
out.close()
