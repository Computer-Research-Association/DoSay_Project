"""행동 집합 생성 함수 모음.

**추가 전용(append-only) 파일이다.** 여기 있는 함수의 동작은 절대 바꾸지 않는다.
이미 학습된 버전이 이 함수로 행동 인덱스를 만들었으므로, 동작이 바뀌면
저장된 체크포인트의 로짓 순서가 통째로 어긋난다.

다른 행동 집합이 필요하면 **새 함수를 추가**하고, 버전 폴더의 env.py 에서
그 함수를 골라 import 한다. 어떤 버전이 무엇을 쓰는지는 각 env.py 의
import 줄에 드러난다.
"""

from game.action import Action


def get_all_action(rows: int, cols: int) -> list[Action]:
    """가능한 모든 직사각형 행동. 1x1 은 제외한다. (9x18 기준 7533개)"""
    actions = []
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    if r1 == r2 and c1 == c2: continue  #  1x1 사이즈인 경우 제외
                    actions.append(Action((r1, c1), (r2, c2)))
    return actions
