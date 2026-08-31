#!/bin/bash
# 500 시드 테스트케이스 생성 — 각 시드 배포판(worst-어닐링) max-of-5, 3000 iter.
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python

# 러너 (seed instance → "seed score")
cat > testcases/runner.sh <<'RUN'
#!/bin/bash
s=$1; i=$2
sc=$(/Users/ball103/DoSay/.venv/bin/python /Users/ball103/DoSay/algorithm/experiment/anneal.py \
    --version Anneal_V01b --seed "$s" --instance "$i" --iters 3000 --neighbor worst --quiet 2>/dev/null)
echo "$s $sc"
RUN
chmod +x testcases/runner.sh

# 작업목록: 500시드 × 5인스턴스
> testcases/jobs.txt
for s in $(seq 100000 100499); do for i in 0 1 2 3 4; do echo "$s $i"; done; done > testcases/jobs.txt
echo "작업 $(wc -l < testcases/jobs.txt)개 시작 $(date +%H:%M)"

# 병렬 실행
> testcases/raw_scores.txt
cat testcases/jobs.txt | xargs -P 10 -L1 bash testcases/runner.sh >> testcases/raw_scores.txt

# 집계 → testcases.json
$PY -c "
import sys; sys.path.insert(0,'.')
import numpy as np, json
from algorithm.experiment.anneal import _make_grid
best={}
for line in open('testcases/raw_scores.txt'):
    p=line.split()
    if len(p)!=2: continue
    try: s=int(p[0]); sc=int(p[1])
    except: continue
    best[s]=max(best.get(s,0), sc)
rows=[]
for s in sorted(best):
    g=_make_grid(s); tot=int(g.sum()); mx=162 if tot%10==0 else 161
    rows.append({'seed':s,'sum':tot,'max_possible':mx,'deploy_score':best[s],'gap':mx-best[s]})
json.dump(rows, open('testcases/testcases.json','w'), indent=1)
sc=np.array([r['deploy_score'] for r in rows])
print(f'완료 {len(rows)}개 | 평균 {sc.mean():.1f} min {sc.min()} max {sc.max()} std {sc.std():.1f} | 만점 {int((sc==162).sum())}개')
"
echo '=== ALL DONE $(date +%H:%M) ==='
