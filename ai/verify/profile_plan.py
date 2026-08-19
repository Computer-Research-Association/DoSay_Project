"""`plan()` 한 판의 시간이 **실제 장치에서** 어디로 가는지 잰다 (2026-08-18).

    python ai/verify/profile_plan.py --checkpoint <ckpt> --device cuda

──────────────────────────────────────────────────────────────────────────
왜 또 프로파일인가
──────────────────────────────────────────────────────────────────────────
지금까지 쓰던 "관측 인코딩 36% + 가치망 60%" 는 **CPU · 폭 32** 에서 잰 값이다.
그 값을 믿고 최적화했더니 실제로는 이렇게 나왔다 (100판, 2070 Laptop):

    첫 빔 (자식 8,192개)   21.4s   -> 12.05s   x1.78
    복구  (자식   512개)   0.834s/회 -> 0.618s/회   x1.35

**같은 최적화가 폭에 따라 다르게 먹힌다.** 작은 배치에서는 계산량이 아니라 커널
실행 대기가 지배한다는 뜻이고, 그렇다면 다음에 고칠 것이 완전히 달라진다.
docs §4.5 가 "병목을 세 번 짚어 두 번 틀렸다" 고 적어 둔 것이 이 함정이다.

──────────────────────────────────────────────────────────────────────────
읽는 법
──────────────────────────────────────────────────────────────────────────
`평균` 열(호출 한 번의 시간)이 핵심이다.

  - 폭이 커질 때 `평균` 이 **비례해서 커지면** 그 구간은 **계산에 묶여 있다**.
    -> 일을 줄이는 최적화(희소화, 재사용)가 먹힌다.
  - 폭이 커져도 `평균` 이 **거의 그대로면** 그 구간은 **실행 대기에 묶여 있다**.
    -> 일을 줄여도 안 빨라진다. 배치를 키우거나 커널 수를 줄여야 한다.

시간은 구간마다 동기화해서 잰다. 비동기 실행이 겹치는 몫을 잃는 대신 **어느
구간인지**를 정확히 얻는다 (`time.time()` 만 쓰면 동기화 지점에 시간이 몰린다).
그래서 여기 합계는 실제 `plan()` 보다 조금 크게 나올 수 있다.
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
from ai.training import resolve_device
from game.board import Board

ROWS, COLS = 9, 18


def make_sync(device: str):
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return getattr(torch.mps, "synchronize", lambda: None)
    return lambda: None


class Profiler:
    """구간별 **배타** 시간. 중첩 호출(observation 안의 legal_mask)을 빼고 잰다."""

    def __init__(self, sync):
        self.sync = sync
        self.stats: dict[str, list] = {}
        self.stack: list[float] = []
        self.patched: list = []

    def wrap(self, obj, attr: str, label: str) -> None:
        original = getattr(obj, attr)

        def inner(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            self.stack.append(0.0)
            try:
                return original(*args, **kwargs)
            finally:
                self.sync()
                spent = time.perf_counter() - start
                inside = self.stack.pop()
                if self.stack:
                    self.stack[-1] += spent         # 부모의 '자식 시간' 에 더한다
                row = self.stats.setdefault(label, [0, 0.0])
                row[0] += 1
                row[1] += spent - inside            # 배타 시간

        setattr(obj, attr, inner)
        self.patched.append((obj, attr, original))

    def restore(self) -> None:
        for obj, attr, original in self.patched:
            setattr(obj, attr, original)
        self.patched.clear()

    def report(self, total: float, header: str) -> None:
        print(f"\n{header}   plan() 벽시계 {total:.2f}s")
        print(f"  {'구간':<26}{'합계(s)':>10}{'비중':>8}{'호출':>9}{'평균(ms)':>11}")
        rows = sorted(self.stats.items(), key=lambda kv: -kv[1][1])
        measured = sum(v[1] for v in self.stats.values())
        for label, (calls, spent) in rows:
            print(f"  {label:<26}{spent:>10.2f}{spent/total:>8.1%}"
                  f"{calls:>9,}{spent/calls*1000:>11.3f}")
        print(f"  {'(그 외 · 파이썬 · 동기화)':<26}{total-measured:>10.2f}"
              f"{(total-measured)/total:>8.1%}")


def run(net, grid, width: int, topk: int, sync) -> tuple[Profiler, float]:
    index = net.index
    prof = Profiler(sync)
    prof.wrap(index, "legal_mask_from_grid", "합법 판정 legal_mask")
    prof.wrap(index, "observation", "관측 인코딩 (마스크 제외)")
    prof.wrap(index, "erase", "자식 만들기 erase")
    prof.wrap(index, "dedup_indices", "중복 제거")
    # 인코더는 가치와 정책이 **함께 쓰는** 가장 비싼 부분이라 따로 잡는다.
    # 나머지 둘은 중첩을 뺀 값이라 각자의 헤드 비용만 남는다.
    prof.wrap(net, "_cells", "인코더 forward (공통)")
    prof.wrap(net, "expected_leftover", "가치 헤드")
    prof.wrap(net, "policy_logits_from_grid", "정책 헤드 + 마스킹")
    sync()
    start = time.perf_counter()
    net.plan(grid, width, topk=topk)
    sync()
    total = time.perf_counter() - start
    prof.restore()
    return prof, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--widths", default="1024,256,64",
                        help="쉼표로. 배포는 첫 빔 1024, 복구 64 를 쓴다")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    device = resolve_device(args.device)
    sync = make_sync(device)
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    model, _env, _info, _search = load_model(source.resolve(), (ROWS, COLS), device=device)
    net = model.policy.q_net
    assert hasattr(net, "plan"), "V15 계열 체크포인트가 필요하다 (plan() 없음)"
    net.value_cache = None                      # 프로파일은 캐시 없이 본다

    grid = torch.as_tensor(Board.from_seed((ROWS, COLS), args.seed).grid,
                           dtype=torch.float32, device=device)
    print(f"체크포인트 {source.name}   장치 {device}   seed {args.seed}   top-{args.topk}")

    net.plan(grid, 32, topk=args.topk)          # 워밍업 (cudnn 알고리즘 선택 등)

    summary = []
    for width in (int(w) for w in args.widths.split(",")):
        prof, total = run(net, grid, width, args.topk, sync)
        prof.report(total, f"=== 폭 {width} ===")
        summary.append((width, total, dict(prof.stats)))

    print("\n=== 폭이 커질 때 호출 한 번의 시간이 어떻게 변하는가 ===")
    print("   (비례해 커지면 계산에 묶임 -> 일을 줄여라."
          " 그대로면 실행 대기에 묶임 -> 배치/커널 수를 줄여라)")
    labels = sorted({k for _, _, s in summary for k in s})
    head = "".join(f"{f'폭 {w}':>14}" for w, _, _ in summary)
    print(f"  {'구간':<26}{head}")
    for label in labels:
        cells = ""
        for _, _, stats in summary:
            calls, spent = stats.get(label, (0, 0.0))
            cells += f"{(spent/calls*1000 if calls else 0):>14.3f}"
        print(f"  {label:<26}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
