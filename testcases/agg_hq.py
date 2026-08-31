"""raw_scores_hq.txt → testcases_hq.json (시드별 max) + 통계. 언제든 실행해 진행 확인."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm.experiment.anneal import _make_grid

best, count = {}, {}
for line in open(ROOT / "testcases" / "raw_scores_hq.txt"):
    p = line.split()
    if len(p) != 2:
        continue
    try:
        s, sc = int(p[0]), int(p[1])
    except ValueError:
        continue
    best[s] = max(best.get(s, 0), sc)
    count[s] = count.get(s, 0) + 1

rows = []
for s in sorted(best):
    g = _make_grid(s); tot = int(g.sum())
    rows.append({"seed": s, "sum": tot, "value_bound": 162 if tot % 10 == 0 else 161,
                 "best_score": best[s], "n_runs": count[s]})
json.dump(rows, open(ROOT / "testcases" / "testcases_hq.json", "w"), indent=1)

sc = np.array([r["best_score"] for r in rows])
runs = np.array([r["n_runs"] for r in rows])
print(f"시드 {len(rows)}개 | 시드당 실행 {runs.min()}~{runs.max()}회 (max-of-N)")
print(f"점수: 평균 {sc.mean():.2f} 중앙 {int(np.median(sc))} min {sc.min()} max {sc.max()} std {sc.std():.1f} | 만점 {int((sc==162).sum())}")
