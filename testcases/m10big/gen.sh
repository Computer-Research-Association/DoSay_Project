#!/bin/bash
# 10의배수 판 1000개 × max-of-16 × 10000 (numba) → 올클리어(162) 비율.
cd /Users/ball103/DoSay
PY=/Users/ball103/DoSay/.venv/bin/python
D=testcases/m10big
INST=16; ITERS=10000

$PY $D/gen_boards.py

cat > $D/runner.sh <<RUN
#!/bin/bash
$PY /Users/ball103/DoSay/$D/run_board.py \$1 $INST $ITERS 2>/dev/null
RUN
chmod +x $D/runner.sh

touch $D/raw.txt
$PY -c "
done=set()
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)==2:
        try: done.add(int(p[0]))
        except: pass
open('$D/todo.txt','w').write('\n'.join(str(i) for i in range(1000) if i not in done))
print('남은:', sum(1 for _ in open('$D/todo.txt')))
"
echo "시작 $(date '+%m-%d %H:%M')"
cat $D/todo.txt | xargs -P 10 -L1 bash $D/runner.sh >> $D/raw.txt

$PY -c "
import numpy as np, json
best={}
for l in open('$D/raw.txt'):
    p=l.split()
    if len(p)!=2: continue
    try: idx,sc=int(p[0]),int(p[1])
    except: continue
    best[idx]=max(best.get(idx,0),sc)
json.dump({int(k):int(v) for k,v in best.items()}, open('$D/result.json','w'))
v=np.array(list(best.values()))
print(f'=== 10의배수 {len(v)}판 (max-of-16 × 10000, numba) ===')
print(f'평균 {v.mean():.2f}  min {int(v.min())}  max {int(v.max())}  std {v.std():.1f}')
print(f'162(올클리어): {int((v==162).sum())}개 ({(v==162).mean()*100:.1f}%)  161={int((v==161).sum())}  160+={int((v>=160).sum())}  158+={int((v>=158).sum())}')
"
echo "=== M10BIG DONE $(date '+%m-%d %H:%M') ==="
