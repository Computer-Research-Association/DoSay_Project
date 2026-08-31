#!/usr/bin/env bash
# 한 판 = 10병렬 어닐링(같은 보드, rng만 다름) → 최고 선택. 2분 예산(≈4000 iter/코어) 반영.
# 사용: par_anneal.sh <neighbor> <iters> <games> [par]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
NB="${1:-single}"; IT="${2:-4000}"; GAMES="${3:-4}"; PAR="${4:-10}"; T0="${5:-3.0}"; BASE="${6:-1234}"
t0=$(date +%s); sum=0
echo "[$NB] iters=$IT, ${PAR}병렬 max/판, ${GAMES}판, T0=$T0"
for g in $(seq 0 $((GAMES-1))); do
  s=$((BASE+g))
  best=$(seq 0 $((PAR-1)) | xargs -P "$PAR" -I{} \
    $PY algorithm/experiment/anneal.py --version Anneal_V01b \
        --seed "$s" --instance {} --iters "$IT" --neighbor "$NB" --T0 "$T0" --quiet \
    | sort -n | tail -1)
  echo "  seed $s : $best"
  sum=$((sum + best))
done
t1=$(date +%s)
avg=$($PY -c "print(f'{$sum/$GAMES:.2f}')")
echo "[$NB] 평균 $avg  (총 $((t1-t0))초, $(( (t1-t0)/GAMES ))초/판)"
