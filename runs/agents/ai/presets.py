"""국소탐색/NRPA 프리셋 표 — 벤치마크와 실기기 봇이 같은 것을 본다.

원래 runs/search_bench.py 안에 있었다. 실기기 봇(runs/play_device.py)도 같은
프리셋을 골라 쓰게 되면서 이리로 옮겼다. 표가 둘로 갈라지면 "벤치마크에서 재던
그 설정" 과 "봇이 실제로 쓴 설정" 이 조용히 달라진다.

각 프리셋에 붙은 주석은 그 설정을 왜 만들었는지에 대한 기록이다. 실험 순서대로
쌓여 있으므로 지우지 말 것.
"""

import numpy as np

from ai.search import localsearch, nrpa

def _table(budget: float, net, value_cache: bool) -> dict[str, tuple]:
    """프리셋 이름 -> (엔진, 설정 kwargs, 한 줄 설명). 신경망 쪽과 대조군을 짝으로 둔다."""
    hand = localsearch.hand_eval(net.index)
    # 2차 배치와 같은 조건을 유지한다 (적응은 3차 프리셋에서만 켠다)
    ls = dict(budget_sec=budget, init_width=1024, init_topk=8,
              repair_width=64, repair_topk=8, ruin_adaptive=False,
              value_cache=value_cache)
    # 첫 빔을 싸게 하고 남는 시간을 전부 탐색에 쓰는 쪽 (21초 -> 3초)
    fast = {**ls, "init_width": 256}

    table = {
        # ── 엔진 1: 국소탐색 ─────────────────────────────────────────────
        # 작은 이웃만 (1차 배치와 같은 설정. 비교 기준으로 남겨 둔다)
        "greedy": ("ls", {**ls, "accept": "greedy", "destroy": "deviate"},
                   "국소탐색 · 작은이웃 · 탐욕"),
        # 큰 이웃 — 1차 배치의 병목("이웃이 작다")을 정면으로 친다
        "ruin-greedy": ("ls", {**ls, "accept": "greedy", "destroy": "ruin"},
                        "국소탐색 · 걷어내기 · 탐욕"),
        "ruin-anneal": ("ls", {**ls, "accept": "anneal", "destroy": "ruin"},
                        "국소탐색 · 걷어내기 · 어닐링"),
        "mixed-anneal": ("ls", {**ls, "accept": "anneal", "destroy": "mixed"},
                         "국소탐색 · 섞기 · 어닐링"),
        "mixed-late": ("ls", {**ls, "accept": "late", "destroy": "mixed"},
                       "국소탐색 · 섞기 · 늦은수락"),

        # ── 3차 배치: 이웃을 더 키운다 ───────────────────────────────────
        # 2차에서 작은이웃 +1.22 -> 섞기 +2.42 로 **이웃 크기가 곧 이득**임이
        # 확인됐다. 그리고 국소탐색(87회)과 NRPA(8,412회)가 97배 다른 작업량으로
        # 같은 134.1 에 멈췄다 — 같은 골짜기에 갇혀 있다는 뜻이다. 골짜기를
        # 넘으려면 한 번에 더 크게 부숴야 한다.
        "ruin-big": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                            "ruin_min": 5, "ruin_max": 20, "ruin_adaptive": False},
                     "국소탐색 · 크게 걷어내기(5~20)"),
        # 막히면 스스로 키운다 (개선되면 되돌아온다)
        "adaptive": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                            "ruin_adaptive": True},
                     "국소탐색 · 적응형 규모"),
        # **크게 부수면 크게 지어야 한다.** 검증(2026-08-15)에서 걷어내기만 5~20 으로
        # 키우고 복구를 그대로 두니 오히려 나빠졌다 (+2/+13 -> +0/+8). 20개를
        # 걷어내 놓고 좁은 빔으로 다시 지으니 제대로 못 짓는 것이다.
        # **파괴 규모와 복구 능력은 짝이므로 같이 키운다.**
        "big-wide": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                            "ruin_min": 5, "ruin_max": 20, "repair_width": 256,
                            "ruin_adaptive": True},
                     "국소탐색 · 크게 걷어내고 크게 짓기"),
        # 복구를 더 잘하면? 반복은 줄지만 한 번의 질이 오른다
        "repair-wide": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                               "repair_width": 256, "ruin_adaptive": True},
                        "국소탐색 · 넓은 복구(W=256)"),
        # 크게 부술 때는 앞쪽을 건드려도 될까 (작은이웃일 때는 앞쪽이 헛수고였다)
        "cut-wide": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                            "cut_lo": 0.15, "ruin_min": 5, "ruin_max": 20,
                            "ruin_adaptive": True},
                     "국소탐색 · 앞쪽까지 + 크게"),
        # ── 4차 배치: **복구를 싸게 해서 반복을 폭발시킨다** ─────────────
        # 3차에서 여섯 설정(작업량 24~8,412회, 350배 차이)이 전부 133.9~134.6 에
        # 몰렸다. 이웃 크기·복구 폭·적응·탐색 계열을 다 바꿔도 안 움직인다.
        #
        # 그런데 알고리즘 팀의 어닐링은 **10초에 137** 이다. 우리는 55초에 134.6 이고
        # 반복이 45회뿐이다. 차이는 한 번의 값이다 — 우리는 복구를 신경망 빔으로
        # 하느라 0.4~1.4초/회를 쓰는데, 어닐링은 싼 수정을 수만 번 한다.
        # **싼 수정 1만 번이 비싼 수정 45번을 이긴다**는 것이 어닐링의 논지다.
        #
        # 그래서 복구 폭을 극단으로 낮춰 반복을 폭발시킨다. 폭 1 은 "가치가 제일
        # 좋게 본 자식을 그냥 따라가기"(= 애프터스테이트 탐욕)이라 신경망 호출이
        # 최소가 된다. 품질은 떨어지지만 횟수로 갚는다.
        "repair-tiny": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                               "repair_width": 4, "cut_lo": 0.15,
                               "ruin_min": 5, "ruin_max": 20, "ruin_adaptive": True},
                        "국소탐색 · 싼 복구(W=4) · 많이"),
        "repair-greedy": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                                 "repair_width": 1, "cut_lo": 0.15,
                                 "ruin_min": 5, "ruin_max": 20, "ruin_adaptive": True},
                          "국소탐색 · 탐욕 복구(W=1) · 아주 많이"),
        # 3차에서 유일하게 움직인 축(앞쪽까지 부수기)을 끝까지 민다
        "cut-all": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                           "cut_lo": 0.0, "ruin_min": 5, "ruin_max": 20,
                           "ruin_adaptive": True},
                    "국소탐색 · 어디든 부수기"),

        # 첫 빔을 절반으로. cut-all 과 같고 init_width 만 512 다 (점수 -1 각오,
        # 시간 -20%). 근거: fast-mixed(W=256) 가 133.32 로 cut-all 보다 1.7 낮다
        "cut-all-512": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                               "cut_lo": 0.0, "ruin_min": 5, "ruin_max": 20,
                               "ruin_adaptive": True, "init_width": 512},
                        "국소탐색 · 어디든 부수기 · 첫 빔 W=512"),

        # 첫 빔을 싸게 (예산 배분 실험)
        "fast-mixed": ("ls", {**fast, "accept": "anneal", "destroy": "mixed"},
                       "국소탐색 · 싼첫빔 · 섞기 · 어닐링"),
        # 대조군 ①: 가치만 손으로. 정책 top-8 가지치기는 여전히 신경망이다
        "hand-mixed": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                              "evaluate": hand}, "손평가 · 섞기 (가치만 대조)"),
        # 대조군 ②: **신경망을 하나도 안 쓴다.** 가치도 손, 가지치기도 없음.
        #   1차 배치에서 대조군의 첫 빔이 130.27 로 예상(122.6)보다 8점 높았는데,
        #   원인이 "손 평가가 신경망 정책의 top-8 가지치기를 쓰고 있어서" 였다.
        #   이것이 오염 없는 진짜 대조군이다.
        "hand-pure": ("ls", {**ls, "accept": "anneal", "destroy": "mixed",
                             "evaluate": hand, "init_topk": 0, "repair_topk": 0},
                      "손평가 · 순수 대조군 (신경망 0)"),

        # ── 엔진 2: NRPA ─────────────────────────────────────────────────
        # **신경망을 싼 자리에만 쓴다.** 빔으로 출발점만 주고(한 번), 롤아웃은
        # 순수 테이블로 빠르게 굴린다.
        #
        # 근거 (2026-08-14 검증, 폭 16, 20초, CPU, seed 1234):
        #     정책사전 O -> 롤아웃 256회, 120점
        #     정책사전 X -> 롤아웃 896회, 126점   <- 5.6배 굴리고 점수도 높다
        # 롤아웃 매 스텝마다 정책망을 부르는 값이 그만큼 비싸고, 우리 정책은
        # 거의 균등이라(policy_kl 1.17) 그 값을 못 한다. 반면 빔 시드는 **판당
        # 한 번**이라 싸고, 131.77 이라는 바닥을 깔아 준다.
        "nrpa-seed": ("nrpa", dict(budget_sec=budget, prior_weight=0.0,
                                   seed_with_beam=True), "NRPA · 빔시드만 (권장)"),
        "nrpa": ("nrpa", dict(budget_sec=budget, prior_weight=1.0,
                              seed_with_beam=True), "NRPA · 빔시드 + 정책사전"),
        "nrpa-noseed": ("nrpa", dict(budget_sec=budget, prior_weight=1.0,
                                     seed_with_beam=False), "NRPA · 정책사전만"),
        # 대조군: 신경망을 하나도 안 쓴 NRPA
        "nrpa-pure": ("nrpa", dict(budget_sec=budget, prior_weight=0.0,
                                   seed_with_beam=False), "NRPA · 순수 (신경망 0)"),

        # ── 기준선 ───────────────────────────────────────────────────────
        "beam-only": ("ls", {**ls, "budget_sec": 0.0}, "빔만 (탐색 없음)"),
    }
    return table


class _NoNet:
    """이름표만 뽑을 때 쓰는 자리끼움.

    hand_eval 은 index 를 클로저에 담아 둘 뿐 부를 때까지 쓰지 않으므로,
    표를 만들어 설명만 읽어 가는 데는 이것으로 충분하다.
    """
    index = None


def build(preset: str, budget: float, net, value_cache: bool = False):
    """프리셋 이름 -> (엔진, 설정, 이름)."""
    table = _table(budget, net, value_cache)
    if preset not in table:
        known = ", ".join(table)
        raise SystemExit(f"모르는 프리셋: {preset}. 고를 수 있는 것: {known}")
    engine, kwargs, label = table[preset]
    cfg = localsearch.Config(**kwargs) if engine == "ls" else nrpa.Config(**kwargs)
    return engine, cfg, label


def preset_names() -> list[str]:
    """고를 수 있는 프리셋 이름. 표에 적힌 순서 그대로."""
    return list(_table(0.0, _NoNet(), False))


def preset_labels() -> dict[str, str]:
    """이름 -> 한 줄 설명. 표를 두 번 적지 않으려고 같은 표에서 뽑는다."""
    return {name: label for name, (_, _, label) in _table(0.0, _NoNet(), False).items()}


def engine_of(preset: str) -> str:
    """'ls' 또는 'nrpa'. 설정을 만들지 않고 엔진만 알고 싶을 때."""
    return _table(0.0, _NoNet(), False)[preset][0]


# ── 수순을 game.Board 좌표로 ─────────────────────────────────────────────────

def actions_to_moves(index, action_ids: list[int]) -> list[tuple[int, int, int, int]]:
    """탐색이 낸 행동 인덱스를 (r1, c1, r2, c2) 로 바꾼다.

    index.r_hi / c_hi 는 **배타적** 경계다. game.Action 의 bottom_right 는
    포함 경계이므로 1을 빼야 한다. (ai/verify/verify_search.py 가 같은 변환을 쓴다)
    """
    return [(int(index.r_lo[a]), int(index.c_lo[a]),
             int(index.r_hi[a]) - 1, int(index.c_hi[a]) - 1) for a in action_ids]
