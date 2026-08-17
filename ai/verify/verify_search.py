"""국소탐색이 낸 수순을 `game.Board` 로 재생해 대조한다.

    python ai/verify/verify_search.py --checkpoint <ckpt> --episodes 3 --budget 20

GPU 에서 만든 수순이 실제 게임에서 성립하는지가 이 프로젝트에서 여러 번 결정적이었다
(V13/V14/V15 전부 이 대조로 버그를 잡았다). 확인하는 것은 넷이다.

    1. 수순의 모든 수가 그 시점에 **엔진 기준으로 합법**인가
    2. 재생한 점수가 국소탐색이 **주장한 점수와 같은가**
    3. 국소탐색 결과가 첫 빔보다 **나빠지지 않았는가** (구조상 그럴 수 없다)
    4. 손 평가 대조군도 같은 검사를 통과하는가
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import torch

from agents.ai.model_loader import load_model
from ai.search import localsearch, nrpa
from ai.training import resolve_device
from game.board import Board

ROWS, COLS = 9, 18
BASE_SEED = 1234


def replay(index, seed: int, actions: list[int]) -> int:
    """엔진으로 수순을 그대로 두어 본다. 불법 수가 있으면 즉시 죽는다."""
    board = Board.from_seed((ROWS, COLS), seed)
    score = 0
    for step, a in enumerate(actions):
        want = ((int(index.r_lo[a]), int(index.c_lo[a])),
                (int(index.r_hi[a]) - 1, int(index.c_hi[a]) - 1))
        match = next((v for v in board.get_valid_actions()
                      if (v.top_left, v.bottom_right) == want), None)
        assert match is not None, (
            f"seed {seed}: {step}번째 수 {want} 가 엔진에서 불법이다")
        _, removed = board.do_action(match)
        score += removed
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--init-width", type=int, default=32)
    parser.add_argument("--repair-width", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    model, env, _info, _search = load_model(source.resolve(), (ROWS, COLS), device=device)
    net = model.policy.q_net
    assert hasattr(net, "plan"), "V15 계열 체크포인트가 필요하다 (plan() 없음)"

    print(f"체크포인트 {source.name}   장치 {device}")
    print(f"{args.episodes}판, 판당 {args.budget:.0f}초, "
          f"첫 빔 W={args.init_width} / 복구 W={args.repair_width}\n")

    base = dict(budget_sec=args.budget, init_width=args.init_width,
                init_topk=8, repair_width=args.repair_width, repair_topk=8)
    nb = dict(budget_sec=args.budget, init_width=args.init_width, init_topk=8,
              iterations=6, rollout_batch=16)
    runs = [
        # 국소탐색 — 파괴 연산자 둘 다 태운다
        ("LS 작은이웃 · 탐욕",
         localsearch, localsearch.Config(**base, accept="greedy", destroy="deviate")),
        ("LS 걷어내기 · 어닐링",
         localsearch, localsearch.Config(**base, accept="anneal", destroy="ruin")),
        ("LS 섞기 · 늦은수락",
         localsearch, localsearch.Config(**base, accept="late", destroy="mixed")),
        # 신경망을 하나도 안 쓴다 — 가치도 손, top-k 가지치기도 끔.
        # `base` 의 init_topk/repair_topk 를 **덮어써야** 하므로 dict 병합으로 만든다.
        ("LS 순수 대조군",
         localsearch, localsearch.Config(**{**base, "init_topk": 0, "repair_topk": 0},
                                         accept="anneal", destroy="mixed",
                                         evaluate=localsearch.hand_eval(net.index))),
        # NRPA — 빔 시드 있는 것과 없는 것, 그리고 신경망을 하나도 안 쓴 것
        ("NRPA 빔시드+정책사전", nrpa, nrpa.Config(**nb, seed_with_beam=True,
                                                  prior_weight=1.0)),
        ("NRPA 정책사전만", nrpa, nrpa.Config(**nb, seed_with_beam=False,
                                              prior_weight=1.0)),
        ("NRPA 순수 (신경망 0)", nrpa, nrpa.Config(**nb, seed_with_beam=False,
                                                   prior_weight=0.0)),
    ]

    print(f"{'설정':<24}{'seed':>6}{'시작':>6}{'탐색후':>8}{'이득':>6}"
          f"{'작업':>8}{'재생':>6}")
    ok = True
    for label, module, cfg in runs:
        for i in range(args.episodes):
            seed = BASE_SEED + i
            grid = torch.as_tensor(Board.from_seed((ROWS, COLS), seed).grid,
                                   dtype=torch.float32, device=device)
            t0 = time.time()
            res = module.solve(net, grid, cfg)
            engine = replay(net.index, seed, res.actions)          # ← 1, 2번 검사
            mark = "O" if engine == res.score else f"X({engine})"
            if engine != res.score:
                ok = False
            if res.init_score and res.score < res.init_score:       # ← 3번 검사
                ok = False
                mark += " !최고가내려감!"
            work = getattr(res, "iterations", None)
            work = work if work is not None else res.rollouts
            print(f"{label:<24}{seed:>6}{res.init_score:>6}{res.score:>8}"
                  f"{res.gain:>+6}{work:>8}{mark:>6}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)

    print()
    if not ok:
        print("불일치가 있다. 국소탐색이 만든 수순을 신뢰할 수 없다.")
        return 1
    print("전부 통과. 수순이 엔진에서 합법이고 점수가 일치한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
