#!/usr/bin/env bash
# DAgger 루프 — 상태수집(on-policy) → 어닐링 라벨링(병렬) → 누적 재학습 → value-beam 평가.
# 사용: bash dl/dagger.sh <rounds> [games_per_round] [label_iters]
#   예: bash dl/dagger.sh 20           # 20라운드 × 300판
#       bash dl/dagger.sh 30 400 800   # 30R, 400판, 라벨 800iter(더 정확·느림)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROUNDS="${1:-20}"; GAMES="${2:-300}"; LABEL_ITERS="${3:-500}"; NSH=10
LOG=dl/dagger_log.txt
echo "=== DAgger 시작: ${ROUNDS}R × ${GAMES}판, 라벨 ${LABEL_ITERS}iter, $(date +%H:%M) ===" | tee -a "$LOG"
for r in $(seq 1 "$ROUNDS"); do
  base=$((20000 + (r - 1) * GAMES))
  echo "----- ROUND $r (base $base, $(date +%H:%M)) -----" | tee -a "$LOG"
  echo "  [1/4] 상태 수집 (on-policy + ε탐험)..."
  $PY dl/gen_states.py --games "$GAMES" --base "$base" --eps 0.25 --sample 0.4 2>&1 | tail -1 | tee -a "$LOG"
  echo "  [2/4] 어닐링 라벨링 (병렬 $NSH)..."
  rm -rf dl/_labelparts
  seq 0 $((NSH - 1)) | xargs -P "$NSH" -I{} "$PY" dl/label_states.py --shard {} --nshards "$NSH" --iters "$LABEL_ITERS" >/dev/null 2>&1
  cat dl/_labelparts/*.jsonl >> dl/value_data.jsonl
  echo "  누적 샘플: $(wc -l < dl/value_data.jsonl)" | tee -a "$LOG"
  echo "  [3/4] 학습..."
  $PY dl/train_value.py --epochs 40 2>&1 | grep 'val MAE' | tail -1 | tee -a "$LOG"
  echo "  [4/4] 평가..."
  $PY dl/eval_value_beam.py --width 5 --depth 3 --games 20 2>&1 | grep -v Warning | tail -1 | sed "s/^/  ROUND $r → /" | tee -a "$LOG"
done
echo "=== 완료 $(date +%H:%M) ===" | tee -a "$LOG"
