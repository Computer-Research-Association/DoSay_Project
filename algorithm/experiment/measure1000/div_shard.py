import sys, os
sys.path.insert(0,'/Users/ball103/DoSay')
import numpy as np
import models.beam as B
from models.board_numba import valid_actions_numba as VA
from models.board import make_board
from algorithm.feature_assistance.math_algorithm import sigmoid

def heuristic_div(grid):
    n9=int((grid==9).sum()); n1=int((grid==1).sum())
    n8=int((grid==8).sum()); n2=int((grid==2).sum())
    f9=0.0 if n9==0 else (-1.0 if n1==0 else -sigmoid(n9/n1,2.5,1.0))
    f8=0.0 if n8==0 else (-1.0 if n2==0 else -sigmoid(n8/n2,2.5,1.0))
    acts=VA(grid); fa=sigmoid(len(acts),0.2,15.0)
    digits=set()
    for (r1,c1,r2,c2) in acts:
        for v in np.unique(grid[r1:r2+1,c1:c2+1]):
            if v>0: digits.add(int(v))
    return f9+f8+fa+len(digits)/9.0

B.heuristic=heuristic_div; B.valid_actions=VA
sh=int(sys.argv[1]); nsh=int(sys.argv[2]); N=1000; S0=1234
VA(make_board(1))
path=f'/Users/ball103/DoSay/algorithm/experiment/measure1000/div_{sh}.txt'
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
    sc=B.play(make_board(seed),5,3)
    out.write(f'{seed} {sc}\n'); out.flush()
out.close()
