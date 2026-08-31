# 탐색 알고리즘 벤치마크 (2026-08)

사과게임(9×18, 합-10 사각형 제거) 최적화를 위한 탐색 알고리즘 비교.
결론: **어닐링(SA)이 실용 천장(~135/162)에 도달하며, 여러 metaheuristic이 같은 곳에 수렴한다.**

## 이 폴더의 벤치 파일 (재사용/레퍼런스)

| 파일 | 내용 |
|------|------|
| `exact_small.py` | 소형판 완전탐색 solver (branch&bound + transposition). `solve_exact(board)` → 증명된 최적. 어닐 근최적성 검증용. |
| `seq_compare.py` | 어닐(worst-point 수순편집) 레퍼런스 + 공용 primitives(`_pick_worst`, `_rollout_tail`, `_fewest`). 다른 벤치 파일이 여기서 import. |
| `ils.py` | Iterated Local Search (완전하강 + kick). |
| `lahc.py` | Late Acceptance Hill Climbing. |
| `search_zoo.py` | Threshold Accepting / Great Deluge / Tabu / VNS. |
| `beam_rollout.py` | rollout(실제 점수) 기반 beam search. |
| `cmp1000/` | 동일 시드 1000판 페어드 비교 결과 (`raw.txt`: seed SA ILS Tabu beam). |

## 핵심 결과 (동일 시드 1000판, 예산 2.5s, 페어드)

| 알고리즘 | 평균 | SA 대비 (95% CI) |
|---|---|---|
| **SA (어닐)** | **128.72** | 기준 (최고) |
| Tabu | 128.35 | −0.375 ± 0.180 |
| ILS | 128.25 | −0.474 ± 0.193 |
| beam | 127.86 | −0.862 ± 0.215 |

- 세 대안 모두 SA보다 **통계적으로 유의하게 낮음**(CI가 0 제외). 단 격차는 미미(<1점).
- n=12에선 "동점~Tabu 우세"로 보였으나 n=1000이 노이즈를 걷어냄 → **표본 크기가 결론을 바꾼 사례**.

## 검증된 결론

- **어닐은 실용 천장의 99%** (수렴 곡선) / 소형판 완전탐색 대비 **98%+** (증명).
- **완전탐색 불가**: 분기 ~27/수 × ~49수 → 경로 10⁷⁰, 중복제거해도 상태 10¹⁹(메모리 벽).
- **여러 패러다임(SA·ILS·Tabu·beam·GA·LAHC·VNS·TA·GD)이 모두 ~같은 천장에 수렴** → 천장은 알고리즘이 아니라 **문제 고유의 한계**.
- 162와의 격차(~24점)는 알고리즘 슬랙이 아니라 **판의 기하적 하한**(완전탐색이 종반 최적성 증명).

## 배포 모델

실제 배포는 `models/anneal.py` (worst-point 수순편집 + Metropolis, 전체-Cython `bench/anneal_cy`). 위 벤치가 이 선택이 측정으로 뒷받침됨을 확인.
