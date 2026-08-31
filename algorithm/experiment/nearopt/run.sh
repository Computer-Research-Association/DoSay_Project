#!/bin/bash
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=algorithm/experiment/nearopt
echo "시작 $(date '+%H:%M')"
seq 0 9 | xargs -P 10 -I{} $PY $D/shard.py {} 10
cat $D/shard_*.txt > $D/raw.txt
$PY -c "
import numpy as np
from collections import defaultdict
by=defaultdict(list)
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)!=7: continue
    sz,seed,cells,opt,an,to,nodes=p
    by[sz].append((int(opt),int(an),int(to),int(cells)))
print('=== 근최적성 곡선: 완전탐색 최적 vs 강한 어닐 ===')
print(f'{\"크기\":<7}{\"셀\":>4}{\"판수\":>4}{\"최적평균\":>8}{\"어닐평균\":>8}{\"비율%\":>7}{\"평균gap\":>8}{\"어닐=최적\":>9}{\"TO\":>4}')
order=['5x9','6x9','7x9']
for sz in order:
    rows=by.get(sz,[])
    if not rows: continue
    valid=[(o,a) for o,a,to,c in rows if to==0]  # 타임아웃 아닌 것만(진짜 최적)
    tos=sum(to for o,a,to,c in rows); cells=rows[0][3]
    o=np.array([x[0] for x in valid]); a=np.array([x[1] for x in valid])
    ratio=(a/o).mean()*100; gap=(o-a).mean(); hit=int((o==a).sum())
    print(f'{sz:<7}{cells:>4}{len(valid):>4}{o.mean():>8.1f}{a.mean():>8.1f}{ratio:>7.1f}{gap:>8.2f}{f\"{hit}/{len(valid)}\":>9}{tos:>4}')
print('\\n(TO=완전탐색 타임아웃, 비율은 진짜최적 확정된 판만)')
"
echo "=== DONE $(date '+%H:%M') ==="
