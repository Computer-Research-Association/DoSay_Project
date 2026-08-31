import sys, json, os
from pathlib import Path
ROOT=Path('/Users/ball103/DoSay'); sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
from algorithm.experiment.experiment import run_model
import numpy as np
from models.board import make_board, apply_move
try:
    from models.board_numba import valid_actions_numba as VA
except: from models.board import valid_actions as VA
RES=ROOT/'algorithm/experiment/measure1000/greedy.json'
res=json.load(open(RES)) if RES.exists() else {}
N=1000; S0=1234
variants=[('nine','Greedy_V04a'),('eight','Greedy_V04b'),('v8c','Greedy_V08c'),
 ('e811','Greedy_V08c811'),('seven','Greedy_V09seven'),('six','Greedy_V09six'),
 ('cluster','Greedy_V09cluster'),('spread','Greedy_V09spread'),('center','Greedy_V09center'),
 ('side','Greedy_V08cside'),('isolated','Greedy_V10isolated'),('weighted','Greedy_V10weighted')]

if __name__=='__main__':
    for key,ver in variants:
        if key in res: continue
        r=run_model(ver,n_games=N,base_seed=S0); res[key]=r['metrics']['avg_score']
        json.dump(res,open(RES,'w'),ensure_ascii=False); print('greedy',key,res[key],flush=True)
    # 랜덤
    if 'random' not in res:
        import random
        VA(make_board(1)); scs=[]
        for i in range(N):
            g=make_board(S0+i).copy(); rng=random.Random(i); sc=0
            while True:
                a=VA(g)
                if not a: break
                sc+=apply_move(g,a[rng.randrange(len(a))])
            scs.append(sc)
        res['random']=round(float(np.mean(scs)),2); json.dump(res,open(RES,'w'),ensure_ascii=False)
        print('greedy random',res['random'],flush=True)
    print('GREEDY DONE',flush=True)
