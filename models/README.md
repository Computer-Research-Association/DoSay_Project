# 사과게임 알고리즘 모델 모음

9×18 격자(값 1-9)에서 **합이 정확히 10인 사각형**을 지워 **총 제거 칸 수(최대 162)**를 최대화하는 문제.
접근법별 대표(최선) 버전을 정리한 폴더. 모든 모델은 `board.py` 유틸을 공유하며 **자급자족(numpy만 필요)**.

## 성능 비교 (랜덤 100판 평균)

| 모델 | 대표 설정 | 평균 점수 | 한 줄 |
|---|---|---|---|
| `greedy.py` | nine+eight+action_count | ~115 | 한 수 앞만 봄(근시안) |
| `beam.py` | width 5, depth 3 (+diversity) | ~119 | heuristic 근사 평가 + 그리디 라인에 갇힘 |
| **`anneal.py`** ★ | worst-이웃 SA + 병렬 max | **~135** | **실제 최종점수로 전체 수순을 전역 최적화 (BEST)** |
| `mcts.py` | UCB + rollout, best-value 백업 | < anneal | 결정적 게임엔 부적합, 2배 느림 |

> **참고 — 이론상 천장 ≈ 137.8** (헤비 어닐링 수렴 추정). 배포판(`anneal.py`)은 판별로 천장 대비 **−2**로 near-최적.
> 완전탐색은 경로 수 ≈ 10^55라 물리적으로 불가능.

## 왜 어닐링이 이기나

- **greedy/beam**: 개별 수를 *heuristic 근사*로 판단 + 앞 수 재검토 불가 → 그리디 라인에 갇힘.
- **anneal**: 해 = *수순 전체*. 실제 최종 제거 칸으로 평가하고, 어느 지점이든 갈아끼우며(+온도로 손해 감수) **전역 탐색** → 장기 의존성(1을 아껴 9와 묶기 등)을 잡음.

## 실행

```bash
# 저장소 루트에서 (-m 모듈 실행)
python -m models.greedy  --seed 1234
python -m models.beam    --seed 1234 --width 5 --depth 3
python -m models.anneal  --seed 1234 --iters 2800 --instances 10   # ★ 배포판
python -m models.mcts    --seed 1234 --iters 3000
```

라이브러리로:
```python
from models.board import make_board
from models.anneal import deploy
score, sequence = deploy(make_board(1234), iters=2800, instances=10)
```

## 튜닝 (anneal)

- **빠른 배포**(판당 ~70초): `iters=2800, instances=10` → ~135
- **천장 근접**(판당 ~5분): `iters=12000, instances=20` → ~137.8
- 같은 알고리즘, iters·instances만 조절.

## 파일

```
models/
├── board.py    공유 유틸 (make_board, valid_actions, apply_move, heuristic)
├── greedy.py   그리디
├── beam.py     빔서치
├── anneal.py   어닐링 + 병렬 max  ★ BEST
├── mcts.py     MCTS
└── README.md
```

## 딥러닝 시도 (별도)

value-net / AlphaZero / policy-gradient / action-max 모방 등은 `dl/`에 별도.
**모두 어닐링(135)에 못 미침** — per-move 신호가 value 정밀도보다 약해(신호<잡음),
이 문제는 "학습"보다 "탐색(어닐링)"이 근본적으로 유리함을 데이터로 확인.
