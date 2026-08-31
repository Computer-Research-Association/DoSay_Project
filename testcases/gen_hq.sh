#!/bin/bash
# 고품질 테스트셋 — 500시드 × 20인스턴스 × 12000 iter, max-of-20 (천장 근접).
# 인스턴스 바깥 순서: 패스마다 모든 시드가 +1 인스턴스 → 언제 멈춰도 균등.
# 이어달리기: raw_scores_hq.txt에 append. agg_hq.py로 언제든 현재 최고 집계.
cd /Users/ball103/DoSay
INSTANCES=20
ITERS=12000
SEED_LO=100000
SEED_HI=100499

cat > testcases/runner_hq.sh <<RUN
#!/bin/bash
s=\$1; i=\$2
sc=\$(/Users/ball103/DoSay/.venv/bin/python /Users/ball103/DoSay/algorithm/experiment/anneal.py \\
    --version Anneal_V01b --seed "\$s" --instance "\$i" --iters $ITERS --neighbor worst --quiet 2>/dev/null)
echo "\$s \$sc"
RUN
chmod +x testcases/runner_hq.sh

# 작업목록 (instance 바깥, seed 안쪽)
> testcases/jobs_hq.txt
for i in $(seq 0 $((INSTANCES-1))); do
  for s in $(seq $SEED_LO $SEED_HI); do echo "$s $i"; done
done > testcases/jobs_hq.txt
echo "작업 $(wc -l < testcases/jobs_hq.txt)개 ($INSTANCES인스턴스 × 500시드, $ITERS iter) 시작 $(date '+%m-%d %H:%M')"

> testcases/raw_scores_hq.txt
cat testcases/jobs_hq.txt | xargs -P 10 -L1 bash testcases/runner_hq.sh >> testcases/raw_scores_hq.txt

/Users/ball103/DoSay/.venv/bin/python testcases/agg_hq.py
echo "=== ALL DONE $(date '+%m-%d %H:%M') ==="
