"""**이미 있는 V12a/V12b 체크포인트로**, 학습 없이 "폭"이 얼마를 내는지 잰다.

    python ai/verify/probe_beam.py --checkpoint ai/models/DQN/V12b/models/V12b_DQN_6000000.zip --device cuda

──────────────────────────────────────────────────────────────────────────
왜 이 실험인가
──────────────────────────────────────────────────────────────────────────
손으로 쓴 평가함수(사과 + 2.5 x 합법수)로 **에피소드 전체 빔**을 재보면 폭에 따라
saturate 없이 계속 오른다 (10판): 폭 1 → 115.90, 8 → 118.30, 32 → 120.10,
128 → 122.70. **깊이가 saturate 하는 것(118.2/119.2/119.0/118.9)과 정반대다.**

그런데 V12b 의 가치는 1수 탐욕에서 쓸모가 없었다 (114.29 = V1.0 과 같음).
**그 이유가 형제 서열이었다** — 한 수 차이 나는 판 28개를 줄 세우려면 오차가
형제 간 차이(~2 사과)보다 작아야 하는데 그러지 못했다.

**전체 빔은 요구하는 것이 다르다.** 빔 안의 부분 게임들은 서로 여러 수가 다르고
점수 차이도 크다. `leftover_mae 0.808` 짜리 가치라면 그 정도 비교는 충분히 할 수
있다. 즉 **같은 가중치가 1수 탐욕에서는 무용지물이었는데 폭에서는 쓸모 있을 수
있다** — 학습 없이 몇 분이면 확인된다.

빔 점수는 `-예상 잔여 사과` 하나뿐이다. 가중치를 맞출 필요가 없다 —
최종 점수 = 162 − 잔여 이고 잔여 예측은 이미 절대 척도이므로, 지금까지 몇 개를
먹었든 상관없이 서로 비교된다.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import numpy as np
import torch

from agents.ai.model_loader import load_model
from ai.training import resolve_device

ROWS, COLS = 9, 18
# 벤치마크(runs/measure.py)와 **같은 판**이어야 짝지은 비교가 된다. 판 운이 점수
# 분산의 76% 라, 시드가 다르면 모델 차이가 판 차이에 묻힌다 (game-analysis §4).
BASE_SEED = 1234
RESULT_DIR = ROOT / "results"


@torch.no_grad()
def play_beam(q_net, grid: torch.Tensor, width: int) -> int:
    """폭 width 로 에피소드 전체를 훑고 최고 점수를 돌려준다."""
    index = q_net.index
    grids = grid[None].clone()
    occupied0 = int((grid != 0).sum())
    best = 0

    while True:
        masks = index.legal_mask_from_grid(grids)
        alive = masks.any(dim=1)
        if (~alive).any():
            done_apples = occupied0 - (grids[~alive] != 0).sum(dim=(1, 2))
            best = max(best, int(done_apples.max()))
        if not bool(alive.any()):
            return best
        grids, masks = grids[alive], masks[alive]

        state_idx, action_idx = masks.nonzero(as_tuple=True)
        children = index.erase(grids[state_idx], action_idx)

        # 지운 칸 집합이 같으면 점수도 같다. 폭을 중복에 낭비하지 않는다.
        _, keep = np.unique(children.reshape(children.shape[0], -1).cpu().numpy(),
                            axis=0, return_index=True)
        children = children[torch.as_tensor(keep, device=children.device)]

        # 예상 최종 점수 = 162 - 예상 잔여. 부분 게임끼리 그대로 비교된다.
        leftover = q_net.expected_leftover(index.observation(children))
        order = torch.argsort(leftover)[:width]
        grids = children[order].contiguous()


def save_scores(source: Path, width: int, base_seed: int, scores: list[int],
                elapsed: float) -> Path:
    """measure.py 와 같은 스키마로 판별 점수를 남긴다 (짝지은 비교용)."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    payload = {
        "agent": f"probe_beam / {source.stem} W{width}",
        "source_path": str(source),
        "beam_width": width,
        "episodes": len(scores),
        "base_seed": base_seed,
        "avg_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores)), 2),
        "best_score": int(np.max(scores)),
        "worst_score": int(np.min(scores)),
        "elapsed_sec": round(elapsed, 1),
        "sec_per_episode": round(elapsed / len(scores), 3),
        "measured_at": now.isoformat(timespec="seconds"),
        "scores": [int(s) for s in scores],
    }
    stamp = now.strftime("%Y%m%d-%H%M%S")
    path = RESULT_DIR / f"probe_{source.stem}_W{width}_seed{base_seed}_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--widths", type=int, nargs="*", default=[1, 8, 32, 128])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED,
                        help="벤치마크와 같은 판이어야 짝지은 비교가 된다 (기본 1234)")
    parser.add_argument("--use-plan", action="store_true",
                        help="아래 play_beam 대신 체크포인트의 q_net.plan() 을 쓴다. "
                             "V15 계열은 plan() 이 정책으로 가지를 좁히므로(POLICY_TOPK) "
                             "이 옵션이 없으면 그 효과가 측정되지 않는다.")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    base_seed = args.base_seed

    device = resolve_device(args.device)
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    model, env, _info, _search = load_model(source, (ROWS, COLS), device=device)
    q_net = model.policy.q_net
    if not hasattr(q_net, "expected_leftover"):
        raise SystemExit("V12a/V12b 체크포인트가 필요합니다 (애프터스테이트 가치 헤드).")

    if args.use_plan and not hasattr(q_net, "plan"):
        raise SystemExit("이 체크포인트에는 plan() 이 없습니다 (--use-plan 을 빼세요).")
    beam = ((lambda net, grid, width: net.plan(grid, width)[1]) if args.use_plan
            else play_beam)

    print(f"체크포인트 : {source.name}   장치: {device}")
    print(f"{args.episodes}판, seed {base_seed}~{base_seed + args.episodes - 1}")
    if args.use_plan:
        mod = sys.modules[type(q_net).__module__]
        print(f"빔 = 체크포인트의 plan()   MODE={getattr(mod, 'MODE', '?')}  "
              f"POLICY_TOPK={getattr(mod, 'POLICY_TOPK', 0)}")
    print("빔 점수 = -예상 잔여 사과 (가중치 없음)\n")
    print(f"{'폭':>6}{'점수':>9}{'std':>8}{'판당':>9}")

    for width in args.widths:
        start = time.time()
        scores = []
        for i in range(args.episodes):
            env.reset(options={"board_source": base_seed + i})
            grid = torch.as_tensor(env.unwrapped.board.grid, dtype=torch.float32, device=device)
            scores.append(beam(q_net, grid, width))
            # 폭이 크면 한 폭에 한 시간이 넘는다. 살아 있는지 알 수 있어야 한다.
            done = i + 1
            if args.progress_every and (done == 1 or done % args.progress_every == 0):
                rate = (time.time() - start) / done
                print(f"    [W={width} {done:>4}/{args.episodes}] 평균 {np.mean(scores):6.2f}  "
                      f"(판당 {rate:.1f}s, 남은 시간 약 {int(rate * (args.episodes - done))}s)",
                      flush=True)
        elapsed = time.time() - start
        print(f"{width:>6}{np.mean(scores):>9.2f}{np.std(scores):>8.2f}"
              f"{elapsed / args.episodes:>8.1f}s", flush=True)
        # **판별 점수를 남긴다.** 평균만 있으면 짝지은 비교를 못 한다 — 같은 100판에서
        # +1~2 점 차이는 짝을 안 지으면 SE 2.1 에 묻힌다 (game-analysis §4).
        print(f"       -> {save_scores(source, width, base_seed, scores, elapsed).name}",
              flush=True)

    print("""
읽는 법
  폭을 키울수록 오르면  -> 같은 가중치가 1수 탐욕에서는 못 쓰였어도 **폭에서는
                         쓸모 있다**. 그러면 V13(롤아웃)보다 이쪽이 싸고 강할 수 있다.
  폭 1 과 별 차이 없으면 -> 이 가치는 부분 게임 비교에도 못 쓴다. 롤아웃이 답이다.

참고 (100판 실측): V1.0 114.49 / 1수앞규칙 116.49 / V12b 1수탐욕 114.29 /
                  V9dB 빔서치 121.45 / 어닐링 ~137
주의: 판 수가 적으면 앞쪽 시드가 쉬워 절대값이 높게 나온다
      (앞 8판 +4.6, 앞 10판 +2.4, 앞 20판 +2.6).""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
