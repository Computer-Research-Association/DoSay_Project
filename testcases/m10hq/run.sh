#!/bin/bash
# 10의 배수 판 100개 × 20인스턴스 × 12000 iter (HQ와 동일 강도) → 올클리어(162) 비율.
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=testcases/m10hq
INSTANCES=20
ITERS=12000

$PY $D/gen_boards.py

cat > $D/runner.sh <<RUN
#!/bin/bash
$PY /Users/ball103/DoSay/$D/run_one.py \$1 \$2 $ITERS 2>/dev/null
RUN
chmod +x $D/runner.sh

# 작업목록: instance 바깥, index 안쪽 (패스마다 전 판 +1 인스턴스)
> $D/jobs.txt
for i in $(seq 0 $((INSTANCES-1))); do
  for idx in $(seq 0 99); do echo "$idx $i"; done
done > $D/jobs.txt
echo "작업 $(wc -l < $D/jobs.txt)개 ($INSTANCES×100판, $ITERS iter) 시작 $(date '+%m-%d %H:%M')"

> $D/raw.txt
cat $D/jobs.txt | xargs -P 10 -L1 bash $D/runner.sh >> $D/raw.txt

# 집계 — 162 비율
$PY -c "
import numpy as np, json
best={}
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)!=2: continue
    try: idx,sc=int(p[0]),int(p[1])
    except: continue
    best[idx]=max(best.get(idx,0),sc)
v=np.array(list(best.values()))
json.dump({int(k):int(x) for k,x in best.items()}, open('$D/result.json','w'), indent=1)
print(f'=== 10의배수 판 {len(v)}개 (max-of-20 × 12000) ===')
print(f'평균 {v.mean():.1f}  min {int(v.min())}  max {int(v.max())}  std {v.std():.1f}')
print(f'162(올클리어): {int((v==162).sum())}개 ({(v==162).mean()*100:.0f}%)')
print(f'161={int((v==161).sum())} 160+={int((v>=160).sum())} 158+={int((v>=158).sum())}')
print('점수분포 상위:', sorted(v.tolist())[::-1][:15])
"
echo '=== M10HQ DONE ==='
