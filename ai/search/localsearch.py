"""해공간 국소탐색 — **빔이 낸 수순을 부수고 신경망 빔으로 다시 짓는다.**

    1. 빔으로 한 판을 푼다                      -> 131.77 (폭 1024 + top-8, 25초)
    2. 수순의 한 지점을 골라 **일부러 다른 수**를 둔다
    3. 그 뒤를 빔으로 다시 짓는다                -> 새 해답
    4. 받아들일지 정한다 (탐욕 / 어닐링 / 늦은수락)
    5. 예산이 끝날 때까지 2~4 반복. 최고를 낸다

──────────────────────────────────────────────────────────────────────────
왜 이 계열인가
──────────────────────────────────────────────────────────────────────────
**빔은 되돌릴 수 없다.** 5수째에 버린 가지가 실은 정답으로 가는 길이었어도 끝이다.
폭을 키우면 살아남을 확률이 오르지만 필요한 폭이 지수적으로 는다 (배가당
+1.54 -> +1.27 -> ...). 알고리즘 팀이 같은 게임에서 빔 120 -> 어닐링 137 로
**+17** 을 얻은 것이 이 차이다.

국소탐색은 접두사 트리를 훑지 않는다. **완성된 수순을 갖고 중간을 고친다.**

**실측 (2026-08-13, 폭 32, 6판).** 빔의 수순을 6수마다 한 번씩, 가치가 2등으로
본 수로 비틀고 나머지를 다시 풀었더니:

    빔 평균 127.67  ->  한 번만 비틀어도 130.50   (+2.83)
    6판 중 5판 개선, 시도한 지점의 12.1% 가 개선

가능한 (지점, 대안) 조합이 판당 약 1,600개인데 **0.6% 만 보고 +2.83** 이다.
빔의 해답은 국소최적에서 한참 멀다.

──────────────────────────────────────────────────────────────────────────
어디를 부술 것인가 — 측정으로 정했다
──────────────────────────────────────────────────────────────────────────
2차 측정(폭 32, 3판, 3수마다 상위 2개)에서 **누적 최고**가 이렇게 움직였다:

    비튼 횟수    1    2    4    8   16   32
    평균 이득  +0.0 +0.0 +0.0 +0.0 +0.0 +2.67

비틀기를 수순 앞쪽(k=0,3,6,...)부터 순서대로 시도했으므로, **앞 16번(수순의 앞
절반)이 전부 헛수고였고 이득은 뒤쪽 절반에서만 나왔다.**

이유는 분명하다. 앞쪽에서 비틀면 빔이 55수를 통째로 다시 짜는데, 그 공간은 이미
원래 빔이 잘 훑은 곳이라 거의 같은 자리로 돌아온다. 뒤쪽에서 비틀면 빔이 **끝내기
단계에 굳어 버린 선택**을 다시 본다.

그래서 `cut_lo`/`cut_hi` 로 자르는 지점을 **뒤쪽 절반**에 몰아 둔다. 이득이 나는
곳이면서 동시에 **복구가 짧아 싸다** — 일석이조다.

──────────────────────────────────────────────────────────────────────────
신경망이 어디서 일하는가 (트랙 경계)
──────────────────────────────────────────────────────────────────────────
**완성된 해답의 점수는 신경망이 필요 없다** (162 - 잔여 사과, 정확히 셀 수 있다).
알고리즘 팀의 어닐링이 사소한 손 기준으로 137 을 내는 이유가 이것이다.

신경망이 값을 하는 곳은 **부분 해답**이다. 절반쯤 지운 판에서 "앞으로 얼마나 더
먹을 수 있나" 는 계산할 수 없고, 거기가 우리 가치망의 자리다. 그래서 이 파일에서

    국소탐색  : "어디를 부술까" 만 정한다        (아래 고리, 수십 줄)
    신경망 빔 : 부순 자리를 처음부터 다시 짓는다  (실제 계산의 95%)

`Config.evaluate` 에 `hand_eval()` 을 넣으면 **같은 탐색을 손 평가로** 돌린
대조군이 된다. 그 차이가 AI 트랙의 기여분이고, 빔에서 그 값이 **+6.48** 이었다.
**점수만 보고 판단하면 안 된다** — 반드시 둘을 같이 잰다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """국소탐색 한 판의 설정. 기본값은 60초 예산 기준이다."""

    budget_sec: float = 60.0

    # 첫 해답. 폭 1024 + top-8 이 100판 131.77 / 25초 (2026-08-13 실측).
    init_width: int = 1024
    init_topk: int = 8

    # 부순 자리를 다시 짓는 빔. **첫 해답만큼 넓을 필요가 없다** — 국소 수정이라
    # 짧게 끝난다. 좁을수록 반복 횟수가 늘어난다. 이 둘의 균형이 핵심 손잡이다.
    repair_width: int = 64
    repair_topk: int = 8

    # 자르는 지점(수순 길이 대비 비율). **뒤쪽 절반**에 몰아 둔다 — 위 머리말의
    # 2차 측정에서 앞쪽 절반의 비틀기가 전부 헛수고였다.
    cut_lo: float = 0.45
    cut_hi: float = 0.95

    # 그 지점에서 볼 대안 수 (가치 순위 기준, 원래 두던 수는 제외)
    n_alt: int = 4

    # 무엇을 부술 것인가.
    #   deviate : 한 지점의 수 하나를 다른 것으로 바꾸고 뒤를 다시 짓는다 (작은 이웃)
    #   ruin    : 수순에서 m개를 **걷어낸다.** 걷어낸 자리 때문에 불법이 된 뒤쪽 수도
    #             함께 빠지므로 판이 넓게 열린다 (큰 이웃)
    #   mixed   : 둘을 섞는다
    #
    # 1차 배치(2026-08-14)에서 `deviate` 만으로는 +1.25 에서 멈췄고, 반복을 1.8배
    # 늘려도 +0.2 밖에 안 늘었다. **이웃이 작은 것이 병목**이라는 뜻이라 `ruin` 을
    # 넣었다. 알고리즘 팀의 어닐링이 137 을 내는 것도 이웃 구조 때문일 것이다.
    destroy: str = "mixed"
    ruin_min: int = 2
    ruin_max: int = 8
    ruin_prob: float = 0.5       # mixed 에서 ruin 을 고를 확률

    # **막히면 더 크게 부순다.** 개선 없이 `ruin_grow_after` 번 지나갈 때마다
    # 파괴 규모를 한 배씩 키우고, 개선되면 원래대로 돌아온다.
    #
    # 2차 배치(2026-08-14)에서 국소탐색(87회 반복)과 NRPA(8,412 롤아웃)가 **97배
    # 다른 작업량으로 같은 134.1 에 멈췄다.** 둘 다 빔의 해답이라는 같은 골짜기
    # 안에서만 파고 있다는 뜻이다. 규모를 키우는 것이 골짜기를 넘는 방법이다.
    ruin_adaptive: bool = True
    ruin_grow_after: int = 15
    ruin_grow_max: int = 5       # 최대 몇 배까지 키울 것인가

    # "greedy" | "anneal" | "late"
    #   greedy : 더 좋을 때만 받는다 (LNS)
    #   anneal : 가끔 나쁜 것도 받는다. 국소최적 탈출 (어닐링)
    #   late   : late_len 번 전의 해답보다 좋으면 받는다 (늦은수락)
    accept: str = "greedy"
    temp_start: float = 2.0      # 사과 단위. 2 면 "2점 손해" 를 e^-1 확률로 받는다
    temp_end: float = 0.05
    late_len: int = 20

    # 가치 대신 쓸 평가기. None 이면 신경망. 대조군은 `hand_eval(index)` 를 넣는다.
    evaluate: object | None = None

    seed: int = 0

    def __post_init__(self) -> None:
        if self.accept not in ("greedy", "anneal", "late"):
            raise ValueError(f"모르는 수락 규칙: {self.accept}")
        if self.destroy not in ("deviate", "ruin", "mixed"):
            raise ValueError(f"모르는 파괴 연산자: {self.destroy}")
        if not 1 <= self.ruin_min <= self.ruin_max:
            raise ValueError(f"ruin 구간이 이상하다: {self.ruin_min}~{self.ruin_max}")
        if self.ruin_grow_after < 1 or self.ruin_grow_max < 1:
            raise ValueError("ruin 성장 설정이 이상하다")
        if not 0.0 <= self.cut_lo < self.cut_hi <= 1.0:
            raise ValueError(f"자르는 구간이 이상하다: {self.cut_lo}~{self.cut_hi}")


@dataclass
class Result:
    score: int
    actions: list[int]
    init_score: int              # 국소탐색 전, 빔만 썼을 때의 점수
    iterations: int              # 시도한 비틀기 횟수
    accepted: int
    improved: int                # 최고 기록이 갱신된 횟수
    elapsed_sec: float
    init_sec: float = 0.0        # 첫 빔에 쓴 시간. **예산을 여기서 다 쓰면 탐색이 0회다**
    ruins: int = 0               # 큰 이웃(걷어내기)을 쓴 횟수
    deviates: int = 0            # 작은 이웃(수 하나 바꾸기)을 쓴 횟수
    max_grow: int = 1            # 파괴 규모를 최대 몇 배까지 키웠나
    # **구간별 시간.** 병목을 세 번 추측해서 두 번 틀렸다 (np.unique 는 2% 였고,
    # 복구 폭도 아니었다). 추측하지 말고 재도록 남긴다.
    t_destroy: float = 0.0       # 걷어내기 재생 + 대안 고르기 (신경망 아닌 부분)
    t_repair: float = 0.0        # 부순 자리를 다시 짓는 빔 (신경망)
    history: list[int] = field(default_factory=list)   # 최고 점수의 변화

    @property
    def gain(self) -> int:
        return self.score - self.init_score


# ──────────────────────────────────────────────────────────────────────────
# 손 평가 대조군
# ──────────────────────────────────────────────────────────────────────────
def hand_eval(index, w_legal: float = 2.5):
    """학습을 전혀 쓰지 않는 평가기. **AI 기여분을 재는 대조군이 이걸 쓴다.**

    `ai/verify/search_study.py --fullbeam` 이 쓰는 것과 같은 식이다:
    남은 사과가 적을수록, 앞으로 둘 수 있는 수가 많을수록 좋다.
    `plan()` 은 이 값이 **작은** 자식을 고르므로 부호를 그렇게 맞춘다.
    """

    def evaluate(children: torch.Tensor) -> torch.Tensor:
        occupied = (children != 0).flatten(1).sum(dim=1).to(torch.float32)
        legal = index.legal_mask_from_grid(children).sum(dim=1).to(torch.float32)
        return occupied - w_legal * legal

    return evaluate


class _Timing:
    """구간별 누적 시간. 파괴 연산자들이 여기에 적는다."""

    __slots__ = ("destroy", "repair")

    def __init__(self) -> None:
        self.destroy = self.repair = 0.0


# ──────────────────────────────────────────────────────────────────────────
# 수락 규칙
# ──────────────────────────────────────────────────────────────────────────
class _Accepter:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.recent: list[int] = []

    def __call__(self, cand: int, cur: int, progress: float) -> bool:
        cfg = self.cfg
        if cfg.accept == "greedy":
            return cand >= cur
        if cfg.accept == "late":
            # 최근 late_len 번 중 가장 오래된 것과 비교한다. 완만하게 내려갈 수 있어
            # 어닐링처럼 국소최적을 벗어나면서 온도 조절이 필요 없다.
            self.recent.append(cur)
            if len(self.recent) > cfg.late_len:
                self.recent.pop(0)
            return cand >= cur or cand >= self.recent[0]
        # anneal: 기하적으로 식힌다
        if cand >= cur:
            return True
        t = cfg.temp_start * (cfg.temp_end / cfg.temp_start) ** progress
        return bool(self.rng.random() < math.exp((cand - cur) / max(t, 1e-6)))


# ──────────────────────────────────────────────────────────────────────────
# 본체
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def solve(net, grid: torch.Tensor, cfg: Config | None = None) -> Result:
    """판 하나를 국소탐색으로 푼다. `net` 은 V15 계열의 `q_net`.

    반환된 `actions` 는 처음부터 순서대로 둘 수 있는 합법 수순이고,
    `ai/verify/verify_search.py` 가 `game.Board` 로 재생해 점수를 대조한다.
    """
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    index = net.index
    start = time.time()
    occupied0 = int((grid != 0).sum().item())

    # ── 1. 첫 해답 (빔) ──────────────────────────────────────────────────
    path, init_score, _ = net.plan(grid, cfg.init_width, topk=cfg.init_topk,
                                   evaluate=cfg.evaluate)
    init_sec = time.time() - start
    cur_boards = [b for b, _ in path]
    cur_actions = [a for _, a in path]
    cur_score = init_score
    best_actions, best_score = cur_actions, cur_score

    accepter = _Accepter(cfg, rng)
    clock = _Timing()
    history = [best_score]
    iterations = accepted = improved = 0

    # ── 2. 부수고 다시 짓기 ──────────────────────────────────────────────
    ruins = deviates = 0
    since_improve = 0            # 마지막 갱신 이후 몇 번 헛돌았나
    grow_used = 1
    while True:
        elapsed = time.time() - start
        if elapsed >= cfg.budget_sec or len(cur_actions) < 2:
            break

        use_ruin = (cfg.destroy == "ruin" or
                    (cfg.destroy == "mixed" and rng.random() < cfg.ruin_prob))
        if use_ruin:
            # 막힌 만큼 크게 부순다 (개선되면 아래에서 1 로 되돌린다)
            grow = 1
            if cfg.ruin_adaptive:
                grow = min(1 + since_improve // cfg.ruin_grow_after, cfg.ruin_grow_max)
            grow_used = max(grow_used, grow)
            cand = _ruin(net, grid, occupied0, cur_boards, cur_actions, cfg, rng,
                         grow, clock)
            ruins += 1
        else:
            cand = _deviate(net, occupied0, cur_boards, cur_actions, cfg, rng, clock)
            deviates += 1
        iterations += 1
        if cand is None:
            continue
        cand_boards, cand_actions, total = cand

        if accepter(total, cur_score, min(elapsed / cfg.budget_sec, 1.0)):
            accepted += 1
            cur_boards, cur_actions, cur_score = cand_boards, cand_actions, total

        if total > best_score:
            improved += 1
            since_improve = 0
            best_score, best_actions = total, cand_actions
        else:
            since_improve += 1
        history.append(best_score)

    return Result(score=best_score, actions=best_actions, init_score=init_score,
                  iterations=iterations, accepted=accepted, improved=improved,
                  elapsed_sec=time.time() - start, init_sec=init_sec,
                  ruins=ruins, deviates=deviates, max_grow=grow_used,
                  t_destroy=clock.destroy, t_repair=clock.repair, history=history)


@torch.no_grad()
def _pick_alternative(net, board: torch.Tensor, taken: int, cfg: Config,
                      rng: np.random.Generator) -> int | None:
    """이 판에서 원래 두던 수 말고, 평가기가 좋게 본 대안 하나를 뽑는다."""
    index = net.index
    mask = index.legal_mask_from_grid(board[None])[0]
    acts = mask.nonzero(as_tuple=True)[0]
    if acts.numel() <= 1:
        return None

    children = index.erase(board[None].expand(acts.numel(), -1, -1), acts)
    evaluate = cfg.evaluate or net._leftover_chunked
    order = torch.argsort(evaluate(children))
    ranked = [int(a) for a in acts[order].tolist() if int(a) != taken]
    if not ranked:
        return None
    return ranked[int(rng.integers(min(cfg.n_alt, len(ranked))))]


# ──────────────────────────────────────────────────────────────────────────
# 파괴 연산자 — 둘 다 (판들, 수순, 총점) 을 돌려준다
#
# 불변식: boards[i] 는 **수 i 를 두기 전의 판**. 아래 둘 다 이것을 지켜야
# `verify_search.py` 의 엔진 재생이 통과한다.
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _deviate(net, occupied0: int, boards: list, actions: list[int],
             cfg: Config, rng: np.random.Generator, clock=None):
    """**작은 이웃.** 한 지점의 수 하나를 다른 것으로 바꾸고 뒤를 다시 짓는다."""
    n = len(actions)
    lo = max(1, int(n * cfg.cut_lo))
    hi = max(lo + 1, int(n * cfg.cut_hi))
    k = int(rng.integers(lo, min(hi, n)))

    t0 = time.time()
    board = boards[k]
    alt = _pick_alternative(net, board, actions[k], cfg, rng)
    if clock is not None:
        clock.destroy += time.time() - t0
    if alt is None:
        return None

    after = net.index.erase(board[None], torch.as_tensor([alt], device=board.device))[0]
    # `plan()` 은 넘긴 판에서 앞으로 얻을 점수를 돌려준다. 이미 얻은 것은 지워진
    # 칸 수로 정확히 계산된다 (점수 = 처음 사과 - 남은 사과).
    got = occupied0 - int((after != 0).sum().item())
    t0 = time.time()
    tail, rest, _ = net.plan(after, cfg.repair_width, topk=cfg.repair_topk,
                             evaluate=cfg.evaluate)
    if clock is not None:
        clock.repair += time.time() - t0
    return (boards[:k + 1] + [b for b, _ in tail],
            actions[:k] + [alt] + [a for _, a in tail],
            got + rest)


@torch.no_grad()
def _ruin(net, grid: torch.Tensor, occupied0: int, boards: list, actions: list[int],
          cfg: Config, rng: np.random.Generator, grow: int = 1, clock=None):
    """**큰 이웃.** 수순에서 m개를 걷어내고 남은 것을 순서대로 다시 둔다.

    핵심은 **연쇄**다. 어떤 수를 걷어내면 그 칸들이 판에 남고, 그 칸을 지나가던
    뒤쪽 수들이 합 10 을 넘겨 불법이 된다. 그것들도 함께 빠지므로 **판이 넓게
    열린다** — 하나를 걷어냈는데 열 개가 빠지기도 한다.

    ── 재생을 왜 numpy 로 하는가 (2026-08-17) ──────────────────────────────
    4차 배치에서 복구 폭을 64분의 1(W=64 -> 1)로 줄였는데 반복이 1.8배밖에
    안 늘었다. 한 번의 비용 중 80% 가 복구가 아닌 곳에 있다는 뜻이고, 그게 이
    재생 고리였다. 원인은 폭이 아니라 **동기화**다 — 수마다 `bool(gpu_tensor)`
    를 부르면 GPU 가 그때마다 멈춰 선다 (판당 57번).

    판은 9x18 int8 이고 합법 판정은 부분합 + 네 변 확인이 전부라, numpy 로 하면
    수당 수십 마이크로초다. GPU 왕복은 **끝에 한 번**만 한다.
    """
    t0 = time.time()
    index = net.index
    n = len(actions)
    m = int(rng.integers(cfg.ruin_min * grow, cfg.ruin_max * grow + 1))
    m = max(1, min(m, n - 1))

    lo = int(n * cfg.cut_lo)
    pool = np.arange(lo, n) if n - lo >= m else np.arange(n)
    drop = set(rng.choice(pool, size=min(m, len(pool)), replace=False).tolist())

    # 첫 번째로 걷어내는 자리 앞은 아무것도 안 바뀐다. 그 앞은 캐시를 그대로 쓴다.
    first = min(drop)
    rects = _rect_bounds(index)
    start = boards[first] if first < len(boards) else grid
    board_np = start.detach().to(torch.int16).cpu().numpy().copy()

    kept_np: list[np.ndarray] = []
    kept_actions: list[int] = list(actions[:first])
    for i in range(first, n):
        a = actions[i]
        if i in drop:
            continue
        r0, c0, r1, c1 = rects[a]
        sub = board_np[r0:r1, c0:c1]
        # `game.Board` 의 규칙 그대로: 합이 정확히 10, 그리고 네 변이 비어 있지 않다
        if sub.sum() != 10:
            continue                      # 걷어낸 것 때문에 불법이 됐다 (연쇄)
        if not (sub[0].any() and sub[-1].any() and sub[:, 0].any() and sub[:, -1].any()):
            continue
        kept_np.append(board_np.copy())
        kept_actions.append(a)
        sub[:] = 0

    device = grid.device
    tail_boards = ([torch.as_tensor(np.stack(kept_np), dtype=torch.float32, device=device)]
                   if kept_np else [])
    kept_boards = list(boards[:first]) + (list(tail_boards[0]) if tail_boards else [])
    board = torch.as_tensor(board_np, dtype=torch.float32, device=device)

    got = occupied0 - int((board_np != 0).sum())
    if clock is not None:
        clock.destroy += time.time() - t0
    t0 = time.time()
    tail, rest, _ = net.plan(board, cfg.repair_width, topk=cfg.repair_topk,
                             evaluate=cfg.evaluate)
    if clock is not None:
        clock.repair += time.time() - t0
    return (kept_boards + [b for b, _ in tail],
            kept_actions + [a for _, a in tail],
            got + rest)


_RECT_CACHE: dict[int, np.ndarray] = {}


def _rect_bounds(index) -> np.ndarray:
    """행동 -> (r_lo, c_lo, r_hi, c_hi). 한 번만 CPU 로 내린다."""
    key = id(index)
    if key not in _RECT_CACHE:
        _RECT_CACHE[key] = np.stack([
            index.r_lo.cpu().numpy(), index.c_lo.cpu().numpy(),
            index.r_hi.cpu().numpy(), index.c_hi.cpu().numpy()], axis=1)
    return _RECT_CACHE[key]
