"""국소탐색 벤치마크 — 100판 고정, JSON 저장. `RunQueue` 로 줄줄이 걸 수 있다.

    python runs/search_bench.py --checkpoint <ckpt> --preset greedy --device cuda

`--preset` 하나가 실험 하나다. 프리셋을 바꿔 가며 여러 번 걸면 표가 쌓인다.

──────────────────────────────────────────────────────────────────────────
반드시 대조군과 함께 볼 것
──────────────────────────────────────────────────────────────────────────
`hand-greedy` / `hand-anneal` 은 **같은 탐색을 손 평가로** 돌린 것이다. 신경망
프리셋과의 차이가 AI 트랙의 기여분이고, 빔에서 그 값이 **+6.48** 이었다.
**점수만 보면 "탐색이 다 한 것" 과 구분되지 않는다.**

──────────────────────────────────────────────────────────────────────────
판정 기준 (2026-08-13 에 미리 정해 둔 것)
──────────────────────────────────────────────────────────────────────────
    같은 100판(seed 1234~), 판당 60초 이하에서 **131.77 을 넘는가.**
    판당 시간이 60초를 넘으면 점수와 무관하게 실패로 친다.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

import numpy as np
import torch

from agents.ai.model_loader import load_model
from agents.ai.presets import build
from ai.search import localsearch, nrpa
from ai.training import resolve_device
from game.board import Board

ROWS, COLS = 9, 18
BASE_SEED = 1234
RESULT_DIR = ROOT / "results"

# 기준선 (같은 100판, seed 1234~1333)
BEAM_BASELINE = 131.77          # V15@50k + 빔 W=1024 + top-8, 25초/판
BUDGET_LIMIT = 60.0             # 빔 계열 규칙


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", default="greedy")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--budget", type=float, default=55.0,
                        help="판당 국소탐색 예산(초). 첫 빔 시간이 여기 포함된다")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--fp16", action="store_true",
                        help="인코더를 반정밀도로. CUDA(텐서코어)/MPS 공통. "
                             "가치/정책 헤드는 fp32 그대로다")
    parser.add_argument("--prefilter", type=int, default=0, metavar="N",
                        help="2단 평가. 자식을 싼 평가로 폭xN 개까지 추린 뒤 "
                             "가치망에 넣는다. 0 이면 끔 (기본)")
    parser.add_argument("--value-cache", action="store_true",
                        help="같은 판을 두 번 평가하지 않는다. 결과는 완전히 같다 "
                             "(실측 평가의 38.5%% 가 중복). GPU 에서 남는 장사인지 A/B 할 것")
    parser.add_argument("--eval-chunk", type=int, default=0, metavar="N",
                        help="한 번에 평가할 자식 수. 0 이면 모델 기본값(4096)")
    parser.add_argument("--shard", metavar="i/n",
                        help="판을 n 등분해 i 번째만 돈다 (0부터). 프로세스 여러 개로 "
                             "나눠 돌린 뒤 runs/merge_shards.py 로 합친다")
    parser.add_argument("--seed", type=int, default=0, help="탐색의 난수 씨앗")
    parser.add_argument("--budget-limit", type=float, default=BUDGET_LIMIT,
                        help="이 시간을 넘으면 실패로 찍는다. 예산이 병목인지 "
                             "구조가 병목인지 보려고 길게 돌릴 때만 함께 올린다")
    args = parser.parse_args()

    device = resolve_device(args.device)
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    model, _env, info, _search = load_model(source.resolve(), (ROWS, COLS), device=device)
    net = model.policy.q_net
    if not hasattr(net, "plan"):
        raise SystemExit("V15 계열 체크포인트가 필요합니다 (plan() 이 없습니다).")

    # ── 속도 손잡이 (2026-08-18). 기본값은 전부 꺼져 있어 옛 숫자가 그대로 나온다 ──
    if args.fp16:
        if device == "cpu":
            raise SystemExit("--fp16 은 cuda/mps 에서만 의미가 있습니다.")
        net.autocast_dtype = torch.float16
    net.prefilter_mult = args.prefilter
    if args.eval_chunk:
        net.eval_chunk = args.eval_chunk

    shard = None
    if args.shard:
        try:
            part, total = (int(v) for v in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard 는 i/n 꼴이어야 합니다: {args.shard!r}")
        if not 0 <= part < total:
            raise SystemExit(f"--shard 범위가 이상합니다: {args.shard}")
        shard = (part, total)

    engine, cfg, label = build(args.preset, args.budget, net,
                               value_cache=args.value_cache)
    module = localsearch if engine == "ls" else nrpa
    uses_neural = (cfg.evaluate is None if engine == "ls"
                   else bool(cfg.prior_weight) or cfg.seed_with_beam)

    print(f"체크포인트 : {source.name}   장치: {device}")
    print(f"프리셋     : {args.preset}  ({label})   엔진: "
          f"{'국소탐색' if engine == 'ls' else 'NRPA'}")
    print(f"{args.episodes}판, seed {args.base_seed}~{args.base_seed + args.episodes - 1}, "
          f"판당 예산 {cfg.budget_sec:.0f}초")
    if engine == "ls":
        print(f"첫 빔 W={cfg.init_width}/top-{cfg.init_topk}, "
              f"복구 W={cfg.repair_width}/top-{cfg.repair_topk}, "
              f"파괴 {cfg.destroy}(걷어내기 {cfg.ruin_min}~{cfg.ruin_max}개), "
              f"자르는 구간 {cfg.cut_lo:.0%}~{cfg.cut_hi:.0%}, 수락 {cfg.accept}")
    else:
        print(f"레벨 {cfg.level}, 반복 {cfg.iterations}, 롤아웃배치 {cfg.rollout_batch}, "
              f"alpha {cfg.alpha}, 정책사전 {cfg.prior_weight}, "
              f"빔시드 {'O' if cfg.seed_with_beam else 'X'}")
    print(f"신경망 사용: {'O' if uses_neural else 'X (순수 대조군)'}")
    knobs = [f"fp16 {'O' if args.fp16 else 'X'}",
             f"2단평가 {args.prefilter if args.prefilter else 'X'}",
             f"값캐시 {'O' if args.value_cache else 'X'}",
             f"eval_chunk {net.eval_chunk}"]
    if shard is not None:
        knobs.append(f"분할 {shard[0]}/{shard[1]}")
    print("속도 손잡이: " + ", ".join(knobs))
    print(f"기준선     : 빔 단독 {BEAM_BASELINE} (같은 100판)\n", flush=True)

    episode_ids = list(range(args.episodes))
    if shard is not None:
        # 이어서 자르지 않고 번갈아 가른다 — 판마다 난이도가 다르므로 이래야
        # 조각별 평균이 서로 비슷해지고 중간 진행 상황이 읽을 만해진다
        episode_ids = [i for i in episode_ids if i % shard[1] == shard[0]]
    n_total = len(episode_ids)

    scores, inits, iters, init_secs = [], [], [], []
    t_destroy, t_repair, start = [], [], time.time()
    for done, i in enumerate(episode_ids, 1):
        grid = torch.as_tensor(Board.from_seed((ROWS, COLS), args.base_seed + i).grid,
                               dtype=torch.float32, device=device)
        run_cfg = type(cfg)(**{**cfg.__dict__, "seed": args.seed + i})
        res = module.solve(net, grid, run_cfg)
        # 두 엔진의 '얼마나 많이 시도했나' 를 한 이름으로 모은다
        work = res.iterations if engine == "ls" else res.rollouts
        scores.append(res.score); inits.append(res.init_score)
        iters.append(work); init_secs.append(res.init_sec)
        t_destroy.append(getattr(res, "t_destroy", 0.0))
        t_repair.append(getattr(res, "t_repair", 0.0))
        if args.progress_every and (done == 1 or done % args.progress_every == 0):
            rate = (time.time() - start) / done
            unit = "반복" if engine == "ls" else "롤아웃"
            print(f"    [{done:>4}/{n_total}] 평균 {np.mean(scores):6.2f} "
                  f"(시작 {np.mean(inits):6.2f}, 이득 {np.mean(scores)-np.mean(inits):+5.2f}, "
                  f"{unit} {np.mean(iters):.0f}회)  판당 {rate:.1f}s, "
                  f"남은 시간 약 {int(rate * (n_total - done))}s", flush=True)

    elapsed = time.time() - start
    per = elapsed / n_total
    s, b = np.array(scores, float), np.array(inits, float)

    print(f"\n{'':<12}{'점수':>9}{'std':>8}{'판당':>9}")
    print(f"{'빔만':<12}{b.mean():>9.2f}{b.std(ddof=1):>8.2f}")
    print(f"{label:<12}{s.mean():>9.2f}{s.std(ddof=1):>8.2f}{per:>8.1f}s")
    d = s - b
    print(f"\n국소탐색의 이득  {d.mean():+.2f} ± {d.std(ddof=1)/np.sqrt(len(d)):.2f} (SE)"
          f"   개선 {int((d>0).sum())}판 / 악화 {int((d<0).sum())}판")
    print(f"기준선 {BEAM_BASELINE} 대비  {s.mean()-BEAM_BASELINE:+.2f}")
    limit = args.budget_limit
    verdict = ("통과" if s.mean() > BEAM_BASELINE and per <= limit else
               f"예산({limit:.0f}초) 초과 (점수와 무관하게 실패)" if per > limit
               else "기준선 미달")
    print(f"판정: {verdict}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    payload = {
        "agent": f"{engine}/{args.preset} / {source.stem}",
        "engine": engine,
        "source_path": str(source),
        "preset": args.preset,
        "label": label,
        "config": {k: v for k, v in cfg.__dict__.items() if k != "evaluate"},
        "work_unit": "iterations" if engine == "ls" else "rollouts",
        "uses_neural": uses_neural,
        "episodes": n_total,
        "base_seed": args.base_seed,
        # 조각으로 나눠 돌렸을 때 합칠 수 있도록 실제로 돈 판을 남긴다
        "seeds": [args.base_seed + i for i in episode_ids],
        "shard": list(shard) if shard is not None else None,
        "knobs": {"fp16": bool(args.fp16), "prefilter": args.prefilter,
                  "value_cache": bool(args.value_cache),
                  "eval_chunk": int(net.eval_chunk), "device": device},
        "avg_score": round(float(s.mean()), 2),
        "std_score": round(float(s.std(ddof=1)), 2),
        "avg_init_score": round(float(b.mean()), 2),
        "avg_gain": round(float(d.mean()), 2),
        "avg_iterations": round(float(np.mean(iters)), 1),
        "avg_init_sec": round(float(np.mean(init_secs)), 2),
        "avg_destroy_sec": round(float(np.mean(t_destroy)), 2),
        "avg_repair_sec": round(float(np.mean(t_repair)), 2),
        "best_score": int(s.max()), "worst_score": int(s.min()),
        "elapsed_sec": round(elapsed, 1), "sec_per_episode": round(per, 3),
        "budget_limit": args.budget_limit,
        "measured_at": now.isoformat(timespec="seconds"),
        "scores": [int(v) for v in scores],
        "init_scores": [int(v) for v in inits],
    }
    tag = f"_shard{shard[0]}of{shard[1]}" if shard is not None else ""
    path = (RESULT_DIR /
            f"search_{args.preset}_{source.stem}_seed{args.base_seed}{tag}_"
            f"{now.strftime('%Y%m%d-%H%M%S')}.json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
