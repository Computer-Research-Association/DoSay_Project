#!/usr/bin/env bash
# AlphaZero 반복 루프 — 병렬 self-play(CPU) → 학습(MPS) → 평가.
# 사용: bash dl/az/loop.sh <iters> [games_per_iter] [workers] [sims]
#   예: bash dl/az/loop.sh 30            # 30반복 × 80판 × 8워커 × 80sims
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ITERS="${1:-30}"; GPI="${2:-80}"; WORKERS="${3:-8}"; SIMS="${4:-80}"
PER=$(( (GPI + WORKERS - 1) / WORKERS ))     # 워커당 게임 수
LOG=dl/az/loop_log.txt
echo "=== AZ 루프: ${ITERS}반복 × ${GPI}판(워커 ${WORKERS}×${PER}) × ${SIMS}sims, $(date +%H:%M) ===" | tee -a "$LOG"

for it in $(seq 1 "$ITERS"); do
  base=$((30000 + it * 5000))
  echo "----- ITER $it (base $base, $(date +%H:%M)) -----" | tee -a "$LOG"
  echo "  [1/3] self-play (병렬 $WORKERS, CPU)..."
  export PY PER SIMS base it
  seq 0 $((WORKERS - 1)) | xargs -P "$WORKERS" -I{} sh -c '
    "$PY" dl/az/selfplay.py --games "$PER" --sims "$SIMS" \
        --base $((base + 1000 * $1)) --tag "it${it}_$1" --device cpu' _ {} 2>&1 \
    | grep -E "판 avg" | tee -a "$LOG"
  echo "  [2/3] 학습 (MPS)..."
  $PY dl/az/train.py --epochs 4 2>&1 | grep -E "epoch|저장" | tail -2 | tee -a "$LOG"
  echo "  [3/3] 평가..."
  $PY dl/az/eval.py --games 10 --base 1234 --sims 80 --device cpu 2>&1 | grep EVAL \
    | sed "s/^/  ITER $it /" | tee -a "$LOG"
done
echo "=== 완료 $(date +%H:%M) ===" | tee -a "$LOG"
