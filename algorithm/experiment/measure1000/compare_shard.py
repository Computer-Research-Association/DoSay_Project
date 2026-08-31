import sys, os, random, math
sys.path.insert(0,'/Users/ball103/DoSay'); sys.path.insert(0,'/Users/ball103/DoSay/bench')
sys.path.insert(0,'/Users/ball103/DoSay/algorithm/experiment')
import numpy as np
import models.board as BD
from models.board_numba import valid_actions_numba as VA
BD.valid_actions=VA
import models.ga as GA; GA.valid_actions=VA
from models.board import make_board, apply_move, cells_of, TOTAL
from anneal_cy import anneal_cy

def _fewest(g,a): return min(a,key=lambda m:int(np.count_nonzero(g[m[0]:m[2]+1,m[1]:m[3]+1])))
def rollout_val(g,rng,R=1,gp=0.8):
    best=0
    for _ in range(R):
        gg=g.copy(); c=0
        while True:
            a=VA(gg)
            if not a: break
            mv=_fewest(gg,a) if rng.random()<gp else a[rng.randrange(len(a))]; c+=apply_move(gg,mv)
        best=max(best,c)
    return best
def beam_rollout(board,seed,W=4,Cn=6,R=1,budget=2.5):
    import time; rng=random.Random(seed); best=0; t0=time.time()
    while time.time()-t0<budget:
        beam=[(board.copy(),0)]
        while beam and time.time()-t0<budget:
            ch=[]
            for g,cl in beam:
                a=VA(g)
                if not a:
                    best=max(best,cl); continue
                for m in sorted(a,key=lambda m:int(np.count_nonzero(g[m[0]:m[2]+1,m[1]:m[3]+1])))[:Cn]:
                    ng=g.copy(); c=apply_move(ng,m); v=cl+c+rollout_val(ng,rng,R); best=max(best,v); ch.append((ng,cl+c,v))
            if not ch: break
            ch.sort(key=lambda x:-x[2]); beam=[(x[0],x[1]) for x in ch[:W]]
    return best
def mcts(board,seed,sims=150):
    rng=random.Random(seed); g=board.copy(); score=0
    while True:
        a=VA(g)
        if not a: return score
        # 간이 MCTS: 각 후보를 sims/후보 rollout, 최고 평균
        best=a[0]; bv=-1
        per=max(1,sims//len(a))
        for m in a:
            ng=g.copy(); c=apply_move(ng,m); tot=0
            for _ in range(per): tot+=rollout_val(ng,rng)
            v=c+tot/per
            if v>bv: bv,best=v,m
        score+=apply_move(g,best)

sh=int(sys.argv[1]); nsh=int(sys.argv[2]); N=1000; S0=1234
VA(make_board(1))
path=f'/Users/ball103/DoSay/algorithm/experiment/measure1000/cmp_{sh}.txt'
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
    b=make_board(seed); bc=np.ascontiguousarray(b)
    an=max(anneal_cy(bc,5000,seed*10+j)[0] for j in range(8))
    ga_s,_=GA.ga(b.copy(),pop=30,gens=40,rng=random.Random(seed))
    mc=mcts(b,seed,150)
    rb=beam_rollout(b,seed,budget=2.5)
    out.write(f'{seed} {an} {int(ga_s)} {mc} {rb}\n'); out.flush()
out.close()
