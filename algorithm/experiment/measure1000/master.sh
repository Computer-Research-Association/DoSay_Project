#!/bin/bash
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=algorithm/experiment/measure1000
echo "=== START $(date '+%m-%d %H:%M') ==="
echo "[1/3] 그리디 12변형 + 랜덤 (하네스)"; $PY $D/greedy.py
echo "[2/3] 빔 스윕 5config (numba)"; seq 0 9 | xargs -P 10 -I{} $PY $D/beam_shard.py {} 10
echo "[3/3] 비교 anneal/ga/mcts/rolloutbeam"; seq 0 9 | xargs -P 10 -I{} $PY $D/compare_shard.py {} 10
echo "[집계]"
$PY -c "
import json, glob, numpy as np
from collections import defaultdict
g=json.load(open('$D/greedy.json'))
# beam
bc=defaultdict(list)
for f in glob.glob('$D/beam_*.txt'):
    for l in open(f):
        p=l.split()
        if len(p)==3: bc[p[1]].append(int(p[2]))
beam={k:round(float(np.mean(v)),2) for k,v in bc.items()}
# cmp
an=[];ga=[];mc=[];rb=[]
for f in glob.glob('$D/cmp_*.txt'):
    for l in open(f):
        p=l.split()
        if len(p)==5: an.append(int(p[1]));ga.append(int(p[2]));mc.append(int(p[3]));rb.append(int(p[4]))
cmp={'anneal':round(float(np.mean(an)),2),'ga':round(float(np.mean(ga)),2),
     'mcts':round(float(np.mean(mc)),2),'rollout_beam':round(float(np.mean(rb)),2),
     'n':len(an)}
final={'boards':'1234~2233 (1000)','greedy':g,'beam':beam,'compare':cmp}
json.dump(final,open('$D/final.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(final,ensure_ascii=False,indent=1))
"
echo "=== ALL DONE $(date '+%m-%d %H:%M') ==="
