"""NRPA (Nested Rollout Policy Adaptation) — **판마다 정책을 새로 배운다.**

    1. 모든 수에 가중치를 하나씩 붙인다 (처음엔 전부 0)
    2. 그 가중치대로 확률을 매겨 한 판을 끝까지 둔다      <- 롤아웃
    3. 여러 번 굴리고 그중 최고 수순을 기억한다
    4. 최고 수순에 나온 수들의 가중치를 올린다             <- 적응
    5. 2~4 를 반복. 이것을 중첩(nested)해서 층으로 쌓는다

──────────────────────────────────────────────────────────────────────────
왜 이 게임에 NRPA 인가
──────────────────────────────────────────────────────────────────────────
**우리 정책 헤드는 판 사이로 일반화가 안 된다** (2026-08-13 확정: `policy_kl`
이 학습 데이터에서조차 1.17 에서 안 내려갔다). 판 하나만 달라도 정답이 바뀌는
조합 퍼즐이라 "판을 보고 좋은 수를 안다" 가 안 배워진다.

**NRPA 는 그 일반화를 아예 요구하지 않는다.** 가중치는 **지금 푸는 판에서만**
쓰고 다음 판에서는 0 부터 다시 배운다. SameGame·Morpion Solitaire 에서 NRPA 가
기록을 갖고 있는 이유가 이것으로 보인다.

그리고 빔의 구조적 약점도 없다. 빔은 한 번 버린 가지를 되돌릴 수 없는데, NRPA 는
**매번 처음부터 다시 두므로** 어떤 수든 다시 시도된다. 기억이 "남겨둔 가지" 가
아니라 **가중치**에 들어 있기 때문이다.

1차 국소탐색 배치(2026-08-14)가 +1.25 에서 멈춘 이유가 "이웃이 작다" 였는데,
NRPA 는 매번 판 전체를 새로 두므로 그 문제 자체가 없다.

──────────────────────────────────────────────────────────────────────────
신경망이 어디서 일하는가 (AI 기여분)
──────────────────────────────────────────────────────────────────────────
둘이고, 둘 다 껐다 켤 수 있어서 **대조군이 그대로 나온다.**

    prior_weight > 0   롤아웃의 확률에 **정책 헤드의 로짓**을 더한다.
                       NRPA 는 보통 균등에서 출발하는데, 우리 정책이 사전분포를
                       주면 출발점이 낫다. 0 이면 순수 NRPA (대조군)
    seed_with_beam     **빔의 해답(131.77)을 초기 최고 수순으로 넣고** 거기로
                       가중치를 몇 번 밀어 준다. 그러면 NRPA 는 131.77 에서
                       출발하고, 최고를 들고 다니므로 **그 아래로 못 내려간다**

`prior_weight=0, seed_with_beam=False` 가 "신경망을 하나도 안 쓴 NRPA" 이고,
그것과의 차이가 AI 트랙의 순 기여분이다.

──────────────────────────────────────────────────────────────────────────
예산과 배치
──────────────────────────────────────────────────────────────────────────
롤아웃을 **R개씩 lockstep 으로** 굴린다. 판마다 상태가 다르지만 합법 판정·판
지우기가 전부 `RectIndex` 의 배치 연산이라 그대로 묶인다. 파이썬 엔진으로 한
개씩 굴리면 33.8ms/판인데, 배치로 묶으면 그 R분의 1 에 가까워진다.

NRPA 의 적응은 순차적이라 배치가 자연스럽지 않다고 알려져 있는데, **레벨 0 을
"R개 롤아웃 중 최고" 로 두면** 순차성을 지키면서 배치가 된다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Config:
    """NRPA 한 판의 설정."""

    budget_sec: float = 55.0

    level: int = 2               # 중첩 깊이. 2 면 (반복 x 반복 x 롤아웃배치)
    iterations: int = 24         # 각 레벨의 반복 수
    rollout_batch: int = 64      # 레벨 0 에서 한 번에 굴리는 롤아웃 수
    alpha: float = 1.0           # 적응 세기 (Rosin 2011 의 기본값)
    temp: float = 1.0            # 롤아웃 softmax 온도

    # 신경망 (둘 다 0/False 면 순수 NRPA = 대조군)
    prior_weight: float = 1.0    # 롤아웃 확률에 더할 정책 로짓의 무게
    seed_with_beam: bool = True  # 빔의 해답에서 출발할 것인가
    init_width: int = 1024       # 그 빔의 폭
    init_topk: int = 8
    seed_adapts: int = 8         # 빔 수순 쪽으로 가중치를 밀어 주는 횟수

    seed: int = 0

    def __post_init__(self) -> None:
        if self.level < 1:
            raise ValueError("level 은 1 이상이어야 한다")
        if self.rollout_batch < 1:
            raise ValueError("rollout_batch 는 1 이상이어야 한다")


@dataclass
class Result:
    score: int
    actions: list[int]
    init_score: int              # 빔 시드 점수 (시드를 안 쓰면 0)
    rollouts: int
    adapts: int
    elapsed_sec: float
    init_sec: float = 0.0
    history: list[int] = field(default_factory=list)

    @property
    def gain(self) -> int:
        return self.score - self.init_score


class _Budget:
    """예산이 끝나면 재귀를 어디서든 즉시 접는다."""

    def __init__(self, seconds: float):
        self.deadline = time.time() + seconds
        self.rollouts = 0
        self.adapts = 0

    @property
    def expired(self) -> bool:
        return time.time() >= self.deadline


@torch.no_grad()
def _rollout_batch(net, grid: torch.Tensor, weights: torch.Tensor, cfg: Config,
                   gen: torch.Generator) -> tuple[int, list[int]]:
    """R개 롤아웃을 lockstep 으로 굴리고 **그중 최고**를 돌려준다."""
    index = net.index
    device = grid.device
    R = cfg.rollout_batch
    boards = grid[None].expand(R, -1, -1).clone()
    occupied0 = int((grid != 0).sum().item())

    max_steps = index.rows * index.cols
    picks = torch.zeros(max_steps, R, dtype=torch.long, device=device)
    live = torch.zeros(max_steps, R, dtype=torch.bool, device=device)

    steps = 0
    for t in range(max_steps):
        mask = index.legal_mask_from_grid(boards)
        alive = mask.any(dim=1)
        if not bool(alive.any()):
            break

        logits = weights[None].expand(R, -1).clone()
        if cfg.prior_weight:
            # 정책 헤드는 이미 불법수를 -1e8 로 눌러 두므로 그대로 더해도 안전하다.
            logits = logits + cfg.prior_weight * net.policy_logits_from_grid(boards)
        logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(logits / cfg.temp, dim=1)
        # 죽은 줄은 전부 -1e9 라 균등이 된다. 어차피 쓰지 않는다.
        chosen = torch.multinomial(probs, 1, generator=gen).squeeze(1)

        picks[t], live[t] = chosen, alive
        idx = alive.nonzero(as_tuple=True)[0]
        boards[idx] = index.erase(boards[idx], chosen[idx])
        steps = t + 1

    scores = occupied0 - (boards != 0).flatten(1).sum(dim=1)
    best = int(scores.argmax().item())
    seq = [int(picks[t, best].item()) for t in range(steps) if bool(live[t, best])]
    return int(scores[best].item()), seq


@torch.no_grad()
def _adapt(net, grid: torch.Tensor, weights: torch.Tensor, seq: list[int],
           cfg: Config) -> None:
    """최고 수순 쪽으로 가중치를 민다 (softmax 정책의 경사 상승).

    Rosin 2011 그대로다: 고른 수는 alpha 만큼 올리고, 그 국면의 모든 수를
    확률에 비례해 내린다. 불법수는 확률이 0 이라 저절로 빠진다.
    """
    index = net.index
    board = grid.clone()
    for a in seq:
        mask = index.legal_mask_from_grid(board[None])[0]
        logits = weights.clone()
        if cfg.prior_weight:
            logits = logits + cfg.prior_weight * net.policy_logits_from_grid(board[None])[0]
        logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(logits / cfg.temp, dim=0)
        weights.sub_(probs, alpha=cfg.alpha)
        weights[a] += cfg.alpha
        board = index.erase(board[None], torch.as_tensor([a], device=board.device))[0]


def _nrpa(net, grid, level: int, weights: torch.Tensor, cfg: Config,
          budget: _Budget, gen: torch.Generator,
          best: tuple[int, list[int]]) -> tuple[int, list[int]]:
    """중첩 재귀. `best` 는 지금까지의 전역 최고 (시드 포함)."""
    if level == 0:
        budget.rollouts += cfg.rollout_batch
        return _rollout_batch(net, grid, weights, cfg, gen)

    local_score, local_seq = best
    for _ in range(cfg.iterations):
        if budget.expired:
            break
        score, seq = _nrpa(net, grid, level - 1, weights.clone(), cfg, budget, gen,
                           (local_score, local_seq))
        if score >= local_score:
            local_score, local_seq = score, seq
        if budget.expired:
            break
        _adapt(net, grid, weights, local_seq, cfg)
        budget.adapts += 1
    return local_score, local_seq


@torch.no_grad()
def solve(net, grid: torch.Tensor, cfg: Config | None = None) -> Result:
    """판 하나를 NRPA 로 푼다. 반환된 `actions` 는 엔진에서 그대로 둘 수 있다."""
    cfg = cfg or Config()
    device = grid.device
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    start = time.time()

    n_actions = int(net.index.r_lo.numel())
    weights = torch.zeros(n_actions, device=device)

    # ── 빔으로 출발점을 만든다 (선택) ────────────────────────────────────
    init_score, init_seq, init_sec = 0, [], 0.0
    if cfg.seed_with_beam:
        path, init_score, _ = net.plan(grid, cfg.init_width, topk=cfg.init_topk)
        init_seq = [a for _, a in path]
        init_sec = time.time() - start
        # 빔의 수순 쪽으로 가중치를 밀어 두면 롤아웃이 그 근처에서 시작한다.
        for _ in range(cfg.seed_adapts):
            _adapt(net, grid, weights, init_seq, cfg)

    budget = _Budget(max(cfg.budget_sec - (time.time() - start), 0.0))
    best_score, best_seq = init_score, init_seq
    history = [best_score]

    # 예산이 남는 동안 최상위 레벨을 반복한다 (한 번으로 안 끝나면 이어서 돈다).
    while not budget.expired:
        score, seq = _nrpa(net, grid, cfg.level, weights, cfg, budget, gen,
                           (best_score, best_seq))
        if score > best_score:
            best_score, best_seq = score, seq
        history.append(best_score)

    return Result(score=best_score, actions=best_seq, init_score=init_score,
                  rollouts=budget.rollouts, adapts=budget.adapts,
                  elapsed_sec=time.time() - start, init_sec=init_sec, history=history)
