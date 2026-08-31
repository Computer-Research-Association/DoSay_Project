#!/bin/bash
# 이어달리기 — raw.txt에 append. 남은 작업만 실행 후 집계.
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=testcases/m10hq
done=$(wc -l < $D/raw.txt)
start=$((done - 20))            # 안전 여유(중복 redo는 max라 무해)
[ $start -lt 1 ] && start=1
echo "이어달리기: 완료 $done, 라인 $start 부터 재개 $(date '+%m-%d %H:%M')"
tail -n +$start $D/jobs.txt | xargs -P 10 -L1 bash $D/runner.sh >> $D/raw.txt

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
print('상위15:', sorted(v.tolist())[::-1][:15])
"
echo '=== M10HQ DONE ==='
