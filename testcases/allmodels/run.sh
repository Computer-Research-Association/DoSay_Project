#!/bin/bash
# 7개 모델 × m10big 1000판 → 점수+수순 기록. 10샤드 병렬.
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=testcases/allmodels
echo "시작 $(date '+%m-%d %H:%M')"
seq 0 9 | xargs -P 10 -I{} $PY $D/eval_shard.py --shard {} --nshards 10

# 합치기 + 요약
cat $D/shard_*.jsonl > $D/results.jsonl
$PY -c "
import json, numpy as np
from collections import defaultdict
rows=[json.loads(l) for l in open('$D/results.jsonl')]
models=['random','greedy','beam','mcts','ga','anneal','dl_az']
print(f'=== {len(rows)}판, 모델별 성능 ===')
print(f'{'model':8s} {'평균':>7} {'중앙':>5} {'min':>4} {'max':>4} {'162수':>6}')
for m in models:
    sc=np.array([r['results'][m]['score'] for r in rows if m in r['results']])
    sc=sc[sc>=0]
    if len(sc)==0: continue
    print(f'{m:8s} {sc.mean():7.2f} {int(np.median(sc)):5d} {int(sc.min()):4d} {int(sc.max()):4d} {int((sc==162).sum()):6d}')
"
echo "=== ALLMODELS DONE $(date '+%m-%d %H:%M') ==="
