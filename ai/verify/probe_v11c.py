"""V11c 체크포인트 하나로 **행동 선택 규칙만** 바꿔가며 병목이 어디인지 가린다.

    python ai/verify/probe_v11c.py --checkpoint ai/models/DQN/V11c/models/V11c_DQN_6000000.zip

학습이 필요 없다. 같은 가중치를 그대로 쓰고 "무엇으로 수를 고를지" 만 바꾼다.

──────────────────────────────────────────────────────────────────────────
왜 이 실험이 중요한가
──────────────────────────────────────────────────────────────────────────
V11c 는 보조 목표를 놀랍도록 잘 배웠다.

    aux/look_corr    0.992     "이 수를 두면 합법수가 몇 개 남는가" 를 거의 정확히 안다
    aux/survive_acc  0.890     "이 사과가 끝까지 남을까" 를 89% 맞힌다

그런데 같은 시드 100판에서 Q argmax 는 **109.17점**이고, 신경망이 r=0.99 로
예측하는 그 양(1수앞 합법수)만 보고 두는 한 줄 규칙은 **113.30점**이다.

즉 재료는 다 있는데 학습된 Q 가 그것을 제대로 쓰지 못하고 있을 수 있다.
그렇다면 병목은 '무엇을 보는가'(표현)가 아니라 '어떻게 고르는가'(가치)이고,
다음 버전은 신경망이 아니라 **가치를 배우는 방식**을 바꿔야 한다.

──────────────────────────────────────────────────────────────────────────
비교하는 규칙들
──────────────────────────────────────────────────────────────────────────
  q       Q argmax                                 벤치마크와 같아야 한다 (109.17)
  look    보조 헤드 argmax                          1수앞 규칙에 해당 (기대 ~113)
  wxN     Q 안의 look 항 계수만 N배               Q = base + w*look 이므로
                                                   Q + (N-1)*w*look 과 같다
  vmap    잔존맵이 가장 적게 남긴다고 보는 수         **핵심 실험**

`vmap` 설명. survive 헤드는 칸마다 '끝까지 남을 확률' p_c 를 낸다. 어떤 수를 둔
뒤의 판 s' 에 대해 sum(p_c) 는 **그 수를 두었을 때 최종적으로 남을 사과 수의 기대값**
이다. 점수 = 162 - 남은 사과 수 이므로 sum(p_c) 를 최소화하는 수가 최선이다.
(먹은 사과 수는 양쪽에서 상쇄되므로 따로 더할 필요가 없다.)

이 값은 TD 가 아니라 **몬테카를로 라벨**로 배운 것이고, 스칼라 하나가 아니라
162개의 조밀한 감독을 받는다. TD 로 배운 Q 보다 형제 서열을 잘 매기는지가
V12 의 방향을 가른다. 다만 후보마다 판을 실제로 만들어 인코딩해야 해서
느리다 (판당 몇 초). 그래서 기본 판 수를 따로 둔다.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import numpy as np
import torch

from agents.ai.model_loader import load_model
from ai.training import resolve_device
from game.board import Board

ROWS, COLS = 9, 18
BASE_SEED = 1234
MASK_FILL = -1e8


def _afterstate_leftover(env, q_net, device, legal):
    """각 후보 수를 둔 뒤 판의 '최종 잔여 사과 기대값' sum(p_c). (len(legal),)

    search.py 와 같은 방식으로 env 의 board 를 잠깐 갈아끼워 관측을 얻는다.
    끝나면 반드시 원래 board 로 되돌린다.
    """
    base = env.unwrapped.board
    saved = base.grid.copy()
    try:
        observations = []
        for index in legal:
            child = Board.from_board(saved)
            child.do_action(env.unwrapped.index_to_action[int(index)])
            env.unwrapped.board = child
            observations.append(env.unwrapped._get_obs())

        with torch.no_grad():
            tensor = torch.as_tensor(np.stack(observations), device=device)
            _q, _look, survive_logit, _mask = q_net.heads(tensor)
            probability = torch.sigmoid(survive_logit)             # (N, R, C)
            occupied = tensor[:, 0] < 0.5                          # 0번 평면 = 빈칸
            return (probability * occupied).sum(dim=(1, 2)).cpu().numpy()
    finally:
        env.unwrapped.board = Board.from_board(saved)


def play(env, q_net, device, seed, mode, extra_weight=0.0):
    obs, _ = env.reset(options={"board_source": seed})
    score = steps = 0
    while steps < ROWS * COLS:
        mask_np = env.unwrapped.get_action_mask()
        legal = np.flatnonzero(mask_np)
        if legal.size == 0:
            break

        if mode == "vmap":
            leftover = _afterstate_leftover(env, q_net, device, legal)
            action = int(legal[int(np.argmin(leftover))])
        else:
            with torch.no_grad():
                tensor = torch.as_tensor(obs[None], device=device)
                q, look, _survive, mask = q_net.heads(tensor)
            if mode == "look":
                scores = torch.where(mask[0], look[0],
                                     torch.full_like(look[0], MASK_FILL))
            else:                                   # q 는 이미 마스킹돼 있다
                scores = q[0] + extra_weight * look[0]
            action = int(scores.argmax())

        if not mask_np[action]:
            raise RuntimeError(f"불법 수 선택 (mode={mode}, action={action})")
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        score = info.get("score", score)
        if terminated or truncated:
            break
    return score, steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100,
                        help="빠른 규칙(q/look/wxN)의 판 수")
    parser.add_argument("--vmap-episodes", type=int, default=30,
                        help="vmap 은 후보마다 판을 인코딩해서 느리다")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--multipliers", type=float, nargs="*",
                        default=[3.0, 10.0, 30.0],
                        help="학습된 look_weight 의 몇 배로 바꿔 볼 것인가")
    args = parser.parse_args()

    device = resolve_device(args.device)
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    model, env, _info, _search = load_model(source, (ROWS, COLS), device=device)
    q_net = model.policy.q_net

    if not hasattr(q_net, "heads"):
        raise SystemExit("이 체크포인트에는 보조 헤드가 없습니다 (V11c 전용 진단입니다).")

    learned = float(q_net.look_weight.item())
    print(f"체크포인트      : {source.name}")
    print(f"학습된 look_weight = {learned:.6f}\n")

    plans = [("q   (Q argmax = 벤치마크)", "q", 0.0, args.episodes),
             ("look (보조 헤드 argmax)", "look", 0.0, args.episodes)]
    for m in args.multipliers:
        plans.append((f"look_weight x{m:g}", "q", learned * (m - 1.0), args.episodes))
    plans.append(("vmap (잔존맵 최소)", "vmap", 0.0, args.vmap_episodes))

    print(f"{'선택 규칙':<30}{'판':>5}{'점수':>8}{'수':>7}{'점/수':>8}{'std':>7}   시간")
    results = {}
    for label, mode, extra, episodes in plans:
        start = time.time()
        scores, steps = [], []
        for i in range(episodes):
            s, n = play(env, q_net, device, BASE_SEED + i, mode, extra)
            scores.append(s)
            steps.append(n)
        mean, moves = float(np.mean(scores)), float(np.mean(steps))
        results[label] = np.array(scores, dtype=float)
        print(f"  {label:<28}{episodes:>5}{mean:>8.2f}{moves:>7.1f}{mean / moves:>8.3f}"
              f"{np.std(scores):>7.1f}   {time.time() - start:.0f}s", flush=True)

    base_label = plans[0][0]
    base = results[base_label]
    print("\n=== Q argmax 대비 짝비교 (같은 판이라 판 운이 상쇄된다) ===")
    for label, arr in results.items():
        if label == base_label:
            continue
        n = min(len(arr), len(base))
        diff = arr[:n] - base[:n]
        se = diff.std(ddof=1) / np.sqrt(n)
        verdict = "유의" if abs(diff.mean()) > 2 * se else "  0 "
        print(f"  {label:<28}{diff.mean():+7.2f} ± {se:.2f}  (n={n})  {verdict}")

    print("""
읽는 법
  look 이나 큰 배수가 Q 를 이긴다  -> TD 가 '남는 선택지' 항의 무게를 너무 작게
                                    잡았다. 그 항을 보상(Phi)으로 되돌리거나
                                    계수를 고정해야 한다.
  vmap 이 Q 를 이긴다              -> 몬테카를로로 배운 조밀한 예측이 TD 로 배운
                                    스칼라 Q 보다 나은 가치함수다. 그러면 V12 는
                                    Q 학습을 버리고 이쪽으로 가야 한다.
  전부 Q 와 같다                   -> 표현이 아니라 정책 자체가 한계. 탐색 교사나
                                    더 긴 학습 쪽으로.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
