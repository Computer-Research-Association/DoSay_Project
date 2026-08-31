#!/usr/bin/env bash
# AlphaZero식 반복 루프 — 라운드마다 어닐링 데이터 추가 → 누적 재학습 → value-beam 평가.
# 사용: bash dl/loop.sh <rounds> [boards_per_round] [iters]
#   예: bash dl/loop.sh 10          # 10라운드, 라운드당 300판
#       bash dl/loop.sh 5 200 1500  # 5라운드, 200판, 어닐링 1500 iter
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROUNDS="${1:-10}"; BPR="${2:-300}"; ITERS="${3:-1200}"
START=6300                 # seeds 6000-6299 는 초기 데이터에 이미 사용됨
DD=dl/_valparts; mkdir -p "$DD"
LOG=dl/loop_log.txt
echo "=== 루프 시작: ${ROUNDS}라운드 × ${BPR}판 (어닐링 ${ITERS} iter) ===" | tee -a "$LOG"

for r in $(seq 1 "$ROUNDS"); do
  lo=$((START + (r-1)*BPR)); hi=$((lo + BPR - 1))
  echo "----- ROUND $r  (seeds $lo-$hi) -----" | tee -a "$LOG"
  rd="$DD/r${r}"; mkdir -p "$rd"
  echo "  [1/3] 데이터 생성..."
  seq "$lo" "$hi" | xargs -P 10 -I{} sh -c "$PY dl/anneal_seq.py --seed {} --iters $ITERS > $rd/{}.jsonl 2>/dev/null"
  cat "$rd"/*.jsonl >> dl/value_data.jsonl          # 누적
  n=$(wc -l < dl/value_data.jsonl)
  echo "  누적 샘플: $n" | tee -a "$LOG"
  echo "  [2/3] 학습..."
  mae=$($PY dl/train_value.py --epochs 40 2>&1 | grep 'val MAE' | tail -1)
  echo "  $mae" | tee -a "$LOG"
  echo "  [3/3] 평가..."
  res=$($PY dl/eval_value_beam.py --width 5 --depth 3 --games 20 2>&1 | grep -v Warning | tail -1)
  echo "  ROUND $r → $res" | tee -a "$LOG"
done
echo "=== 완료. 전체 로그: dl/loop_log.txt ===" | tee -a "$LOG"
