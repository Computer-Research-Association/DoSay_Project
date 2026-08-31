import sys
sys.path.insert(0,'/Users/ball103/DoSay'); sys.path.insert(0,'/Users/ball103/DoSay/bench')
import numpy as np
from models.board import make_board
from anneal_cy import anneal_cy
shard=int(sys.argv[1]); nsh=int(sys.argv[2]); INST=16; ITERS=10000
out=open(f'/Users/ball103/DoSay/bench/big10k_cy/shard_{shard}.txt','w')
for i in range(10000):
    if i%nsh!=shard: continue
    seed=400000+i
    b=np.ascontiguousarray(make_board(seed))
    best=0
    for j in range(INST):
        sc,_=anneal_cy(b, ITERS, seed*100+j)
        if sc>best: best=sc
    out.write(f'{seed} {best}\n'); out.flush()
out.close()
