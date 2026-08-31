#!/bin/bash
# 합≡0 mod10 보드 100개(숫자분포 유지) → 배포판(max-of-5 × 4000) → 올클리어(162) 비율 측정.
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=testcases/m10

mkdir -p $D
# 1) 합≡0 시드 100개 찾기 (200000부터 스캔, 숫자분포는 균등 유지)
$PY -c "
import sys; sys.path.insert(0,'.')
from algorithm.experiment.anneal import _make_grid
seeds=[]; s=200000
while len(seeds)<100:
    if int(_make_grid(s).sum())%10==0: seeds.append(s)
    s+=1
open('$D/seeds.txt','w').write('\n'.join(map(str,seeds)))
print(f'합≡0 시드 {len(seeds)}개 확보 (스캔 {s-200000}개 중, 채택률 {100/(s-200000)*100:.0f}%)')
"

# 2) 작업목록: 100시드 × 5인스턴스
> $D/jobs.txt
while read s; do for i in 0 1 2 3 4; do echo "$s $i"; done; done < $D/seeds.txt > $D/jobs.txt

cat > $D/runner.sh <<'RUN'
#!/bin/bash
s=$1; i=$2
sc=$(/Users/ball103/DoSay/.venv/bin/python /Users/ball103/DoSay/algorithm/experiment/anneal.py \
    --version Anneal_V01b --seed "$s" --instance "$i" --iters 4000 --neighbor worst --quiet 2>/dev/null)
echo "$s $sc"
RUN
chmod +x $D/runner.sh

echo "작업 $(wc -l < $D/jobs.txt)개 시작 $(date +%H:%M)"
> $D/raw.txt
cat $D/jobs.txt | xargs -P 10 -L1 bash $D/runner.sh >> $D/raw.txt

# 3) 집계 — 162 비율
$PY -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from algorithm.experiment.anneal import _make_grid
best={}
for line in open('$D/raw.txt'):
    p=line.split()
    if len(p)!=2: continue
    try: s,sc=int(p[0]),int(p[1])
    except: continue
    best[s]=max(best.get(s,0),sc)
sc=np.array(list(best.values()))
print(f'=== 합≡0 mod10 보드 {len(sc)}개 결과 ===')
print(f'평균 {sc.mean():.1f}  min {sc.min()}  max {sc.max()}  std {sc.std():.1f}')
print(f'162(올클리어): {int((sc==162).sum())}개 ({(sc==162).mean()*100:.0f}%)')
print(f'161+: {int((sc>=161).sum())}개  | 160+: {int((sc>=160).sum())}개')
print('상위10:', sorted(sc.tolist())[-10:])
"
echo '=== M10 DONE ==='
