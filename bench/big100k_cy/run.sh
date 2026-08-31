#!/bin/bash
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=bench/big100k_cy
echo "시작 $(date '+%m-%d %H:%M')"
seq 0 9 | xargs -P 10 -I{} $PY $D/run_shard.py {} 10
cat $D/shard_*.txt > $D/raw.txt
$PY -c "
import numpy as np, json
best={}
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)==2:
        try: best[int(p[0])]=int(p[1])
        except: pass
json.dump(best, open('$D/result.json','w'))
v=np.array(list(best.values()))
print(f'=== 전체Cython 최종 {len(v)}판 (max-of-16 × 10000) ===')
print(f'평균 {v.mean():.3f}  중앙 {int(np.median(v))}  min {int(v.min())}  max {int(v.max())}  std {v.std():.2f}')
print(f'만점162={int((v==162).sum())} ({(v==162).mean()*100:.3f}%)  161={int((v==161).sum())}  160+={int((v>=160).sum())}  150+={int((v>=150).sum())}  140+={int((v>=140).sum())}')
"
echo "=== DONE $(date '+%m-%d %H:%M') ==="
