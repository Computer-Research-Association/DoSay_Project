#!/bin/bash
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=algorithm/experiment/cmp1000
echo "시작 $(date '+%m-%d %H:%M')"
seq 0 4 | xargs -P 5 -I{} $PY $D/shard.py {} 5
cat $D/shard_*.txt > $D/raw.txt
$PY -c "
import numpy as np
A={'SA':[], 'ILS':[], 'Tabu':[], 'beam':[]}
seeds=[]
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)!=5: continue
    seeds.append(int(p[0]))
    A['SA'].append(int(p[1])); A['ILS'].append(int(p[2])); A['Tabu'].append(int(p[3])); A['beam'].append(int(p[4]))
for k in A: A[k]=np.array(A[k])
n=len(seeds)
print(f'=== 동일시드 {n}판 · 페어드 비교 (budget 2.5s) ===')
print(f'{\"알고\":<8}{\"평균\":>8}{\"중앙\":>6}{\"std\":>7}{\"min\":>5}{\"max\":>5}')
for k in ['SA','ILS','Tabu','beam']:
    v=A[k]; print(f'{k:<8}{v.mean():>8.2f}{int(np.median(v)):>6}{v.std():>7.2f}{int(v.min()):>5}{int(v.max()):>5}')
print(f'\\n[SA 대비 페어드 차이]')
sa=A['SA']
for k in ['ILS','Tabu','beam']:
    d=A[k]-sa
    se=d.std()/np.sqrt(n)
    print(f'{k:<8} 평균차 {d.mean():+.3f} ± {1.96*se:.3f}(95%CI)  우세 {int((d>0).sum())} 동점 {int((d==0).sum())} 열세 {int((d<0).sum())}')
"
echo "=== DONE $(date '+%m-%d %H:%M') ==="
