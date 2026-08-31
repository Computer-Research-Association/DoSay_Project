#!/bin/bash
# 가성비판(max-of-16 × 10000, numba)으로 10,000 시드 벤치마크. append 방식(이어달리기 가능).
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=testcases/big10k
INST=16; ITERS=10000
LO=400000; HI=409999

cat > $D/runner.sh <<RUN
#!/bin/bash
$PY /Users/ball103/DoSay/$D/run_board.py \$1 $INST $ITERS 2>/dev/null
RUN
chmod +x $D/runner.sh

# 이미 끝난 시드 제외(이어달리기)
touch $D/raw.txt
$PY -c "
done=set()
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)==2:
        try: done.add(int(p[0]))
        except: pass
with open('$D/todo.txt','w') as f:
    for s in range($LO,$HI+1):
        if s not in done: f.write(str(s)+'\n')
print('남은 시드:', sum(1 for _ in open('$D/todo.txt')))
"
echo "시작 $(date '+%m-%d %H:%M')"
cat $D/todo.txt | xargs -P 10 -L1 bash $D/runner.sh >> $D/raw.txt

$PY -c "
import numpy as np, json
best={}
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)!=2: continue
    try: s,sc=int(p[0]),int(p[1])
    except: continue
    best[s]=max(best.get(s,0),sc)
json.dump({int(k):int(v) for k,v in best.items()}, open('$D/result.json','w'))
v=np.array(list(best.values()))
print(f'=== {len(v)}판 (max-of-16 × 10000, numba) ===')
print(f'평균 {v.mean():.2f}  중앙 {int(np.median(v))}  min {int(v.min())}  max {int(v.max())}  std {v.std():.1f}')
print(f'만점162={int((v==162).sum())} 160+={int((v>=160).sum())} 150+={int((v>=150).sum())}')
"
echo "=== BIG10K DONE $(date '+%m-%d %H:%M') ==="
