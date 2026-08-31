import sys, os
sys.path.insert(0,'/Users/ball103/DoSay'); sys.path.insert(0,'/Users/ball103/DoSay/bench')
import numpy as np
from models.board import make_board
from anneal_cy import anneal_cy
shard=int(sys.argv[1]); nsh=int(sys.argv[2]); INST=16; ITERS=10000; TOTAL=100000
path=f'/Users/ball103/DoSay/bench/big100k_cy/shard_{shard}.txt'
done=set()
if os.path.exists(path):
    for l in open(path):
        p=l.split()
        if len(p)==2:
            try: done.add(int(p[0]))
            except: pass
out=open(path,'a')
for i in range(TOTAL):
    if i%nsh!=shard: continue
    seed=400000+i
    if seed in done: continue
    b=np.ascontiguousarray(make_board(seed))
    best=0
    for j in range(INST):
        sc,_=anneal_cy(b, ITERS, seed*100+j)
        if sc>best: best=sc
    out.write(f'{seed} {best}\n'); out.flush()
out.close()
