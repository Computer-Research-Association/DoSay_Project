#!/bin/bash
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=algorithm/experiment/measure1000
echo "=== START2 $(date '+%m-%d %H:%M') ==="
echo "[1/2] diversity 빔 1000판"; seq 0 9 | xargs -P 10 -I{} $PY $D/div_shard.py {} 10
echo "[2/2] 풀강 어닐(max16x10000) 1000판"; seq 0 9 | xargs -P 10 -I{} $PY $D/annealfull_shard.py {} 10
echo "[집계]"
$PY -c "
import json, glob, numpy as np
f=json.load(open('$D/final.json'))
dv=[int(l.split()[1]) for g in glob.glob('$D/div_*.txt') for l in open(g) if len(l.split())==2]
af=[int(l.split()[1]) for g in glob.glob('$D/annealfull_*.txt') for l in open(g) if len(l.split())==2]
f['beam']['diversity']=round(float(np.mean(dv)),2)
f['anneal_full_deploy']={'avg':round(float(np.mean(af)),2),'n':len(af),'perfect':int((np.array(af)==162).sum())}
json.dump(f,open('$D/final.json','w'),ensure_ascii=False,indent=1)
print('diversity 빔:',f['beam']['diversity'],'(',len(dv),'판)')
print('풀강 어닐:',f['anneal_full_deploy'])
"
echo "=== ALL DONE2 $(date '+%m-%d %H:%M') ==="
