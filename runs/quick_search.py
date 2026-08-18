"""탐색을 원하는 시드에서만 빠르게 돌려 본다. **인자도 TUI 도 없다.**

아래 "여기만 고친다" 칸의 변수만 바꾸고 그냥 실행한다.

    python runs/quick_search.py

`runs/search_bench.py` 와의 차이:

    search_bench.py   100판 고정(seed 1234~1333), 인자로 프리셋을 받는다. **공식 기록용**
    quick_search.py   시드를 손으로 골라 몇 판만. **눈으로 보고 만져 보는 용**

같은 엔진·같은 프리셋 표를 쓰므로 두 스크립트의 숫자는 서로 비교된다. 다만 시드
묶음이 다르면 절대값은 다르다 — 판 운이 점수 분산의 76% 다 (game-analysis.md §4).
**공식 숫자로 쓸 거면 100판(seed 1234~1333)을 돌릴 것.**
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

# ══════════════════════════════════════════════════════════════════════════
# 여기만 고친다
# ══════════════════════════════════════════════════════════════════════════

# 어떤 가중치로 돌릴까
CHECKPOINT = "ai/models/DQN/V15/models/V15_DQN_50000.zip"

# 어떤 판을 풀까. 아무 정수나 몇 개든 된다.
#   빠르게 감 보기      -> [1234, 1235, 1236]
#   공식 100판과 같게   -> list(range(1234, 1334))
#   어려운 판만 (빔이 120 이하였던 것들)
#                      -> [1280, 1276, 1290, 1282, 1256, 1331, 1234 + 42]
SEEDS = [402203, 410533, 412113, 423223, 433993]

# 어떤 탐색을 쓸까. runs/search_bench.py 의 build() 에 있는 이름 그대로.
#   추천              cut-all       국소탐색 최고 (100판 135.00)
#   빔만              beam-only     탐색 없이 첫 빔만 (100판 131.77)
#   NRPA              nrpa-seed     (100판 134.04)
#   대조군 (신경망 0)  nrpa-pure     (100판 129.65)
PRESET = "cut-all"

# 판당 예산(초). 첫 빔 시간이 여기 **포함**된다.
#   빔 W=1024 의 첫 빔이 약 21초다. 55 면 탐색에 34초가 남는다.
BUDGET_SEC = 55.0

DEVICE = "cpu"          # "cuda" | "cpu" | "auto"
SEARCH_SEED = 0          # 탐색의 난수 씨앗. 판마다 여기에 순번이 더해진다
VERIFY_WITH_ENGINE = True   # 낸 수순을 game.Board 로 재생해 점수를 대조 (거의 무료)
SAVE_JSON = True            # results/ 에 남길까

# ══════════════════════════════════════════════════════════════════════════
# 아래는 건드릴 필요 없다
# ══════════════════════════════════════════════════════════════════════════

import json
import time
from datetime import datetime

import numpy as np
import torch

from agents.ai.model_loader import load_model
from ai.search import localsearch, nrpa
from ai.training import resolve_device
from game.board import Board
from search_bench import BEAM_BASELINE, build

ROWS, COLS = 9, 18
RESULT_DIR = ROOT / "results"


def replay_score(index, seed: int, actions: list[int]) -> int:
    """엔진으로 수순을 그대로 두어 본다. 불법 수가 있으면 즉시 죽는다."""
    board = Board.from_seed((ROWS, COLS), seed)
    score = 0
    for step, a in enumerate(actions):
        want = ((int(index.r_lo[a]), int(index.c_lo[a])),
                (int(index.r_hi[a]) - 1, int(index.c_hi[a]) - 1))
        match = next((v for v in board.get_valid_actions()
                      if (v.top_left, v.bottom_right) == want), None)
        if match is None:
            raise AssertionError(f"seed {seed}: {step}번째 수 {want} 가 엔진에서 불법")
        _, removed = board.do_action(match)
        score += removed
    return score


def main() -> int:
    device = resolve_device(DEVICE)
    source = (Path(CHECKPOINT) if Path(CHECKPOINT).is_absolute()
              else ROOT / CHECKPOINT).resolve()
    if not source.exists():
        raise SystemExit(f"체크포인트가 없습니다: {source}\n"
                         "  CHECKPOINT 변수를 고치세요.")

    model, _env, _info, _search = load_model(source, (ROWS, COLS), device=device)
    net = model.policy.q_net
    if not hasattr(net, "plan"):
        raise SystemExit("V15 계열 체크포인트가 필요합니다 (plan() 이 없습니다).")

    engine, cfg, label = build(PRESET, BUDGET_SEC, net)
    module = localsearch if engine == "ls" else nrpa
    uses_neural = (cfg.evaluate is None if engine == "ls"
                   else bool(cfg.prior_weight) or cfg.seed_with_beam)

    print(f"체크포인트 : {source.name}   장치: {device}")
    print(f"프리셋     : {PRESET}  ({label})")
    print(f"신경망 사용: {'O' if uses_neural else 'X (대조군)'}")
    print(f"{len(SEEDS)}판, 판당 예산 {cfg.budget_sec:.0f}초 "
          f"(예상 {len(SEEDS) * cfg.budget_sec / 60:.0f}분)\n")

    print(f"{'seed':>7}{'빔':>6}{'탐색후':>8}{'이득':>6}{'작업':>8}{'초':>7}"
          + ("  엔진" if VERIFY_WITH_ENGINE else ""))
    rows, start = [], time.time()
    for i, seed in enumerate(SEEDS):
        grid = torch.as_tensor(Board.from_seed((ROWS, COLS), seed).grid,
                               dtype=torch.float32, device=device)
        run_cfg = type(cfg)(**{**cfg.__dict__, "seed": SEARCH_SEED + i})
        t0 = time.time()
        res = module.solve(net, grid, run_cfg)
        took = time.time() - t0
        work = res.iterations if engine == "ls" else res.rollouts

        mark = ""
        if VERIFY_WITH_ENGINE:
            got = replay_score(net.index, seed, res.actions)
            mark = "  O" if got == res.score else f"  X({got})"
        print(f"{seed:>7}{res.init_score:>6}{res.score:>8}{res.gain:>+6}"
              f"{work:>8,}{took:>6.0f}s{mark}", flush=True)
        rows.append(dict(seed=seed, score=res.score, init=res.init_score,
                         work=work, sec=round(took, 1),
                         t_destroy=round(getattr(res, "t_destroy", 0.0), 2),
                         t_repair=round(getattr(res, "t_repair", 0.0), 2),
                         init_sec=round(res.init_sec, 2)))

    s = np.array([r["score"] for r in rows], float)
    b = np.array([r["init"] for r in rows], float)
    elapsed = time.time() - start
    per = elapsed / len(SEEDS)

    print(f"\n{'':<10}{'평균':>8}{'std':>7}{'최저':>6}{'최고':>6}")
    print(f"{'빔만':<10}{b.mean():>8.2f}{b.std(ddof=1) if len(b) > 1 else 0:>7.2f}"
          f"{b.min():>6.0f}{b.max():>6.0f}")
    print(f"{'탐색 후':<10}{s.mean():>8.2f}{s.std(ddof=1) if len(s) > 1 else 0:>7.2f}"
          f"{s.min():>6.0f}{s.max():>6.0f}")
    d = s - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    print(f"\n탐색의 이득  {d.mean():+.2f}" + (f" ± {se:.2f} (SE)" if se else "")
          + f"   개선 {int((d > 0).sum())}판 / 악화 {int((d < 0).sum())}판")
    print(f"판당 {per:.1f}초")

    ds, rp = np.mean([r["t_destroy"] for r in rows]), np.mean([r["t_repair"] for r in rows])
    if rp > 0:
        it = np.mean([r["work"] for r in rows])
        print(f"시간 내역    첫 빔 {np.mean([r['init_sec'] for r in rows]):.1f}s / "
              f"부수기 {ds:.1f}s / 다시짓기 {rp:.1f}s   (반복당 {rp / max(it, 1) * 1000:.0f}ms)")

    # 시드 묶음이 다르면 절대값 비교는 의미가 없다. 같을 때만 기준선을 붙인다.
    if list(SEEDS) == list(range(1234, 1334)):
        print(f"공식 100판 기준선 {BEAM_BASELINE} 대비  {s.mean() - BEAM_BASELINE:+.2f}")
    else:
        print(f"(시드 묶음이 공식 100판과 달라 기준선 {BEAM_BASELINE} 과 직접 비교 불가)")

    if SAVE_JSON:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        path = (RESULT_DIR /
                f"quick_{PRESET}_{source.stem}_{len(SEEDS)}seeds_"
                f"{now.strftime('%Y%m%d-%H%M%S')}.json")
        path.write_text(json.dumps({
            "agent": f"quick/{PRESET} / {source.stem}",
            "engine": engine, "preset": PRESET, "label": label,
            "source_path": str(source), "uses_neural": uses_neural,
            "config": {k: v for k, v in cfg.__dict__.items() if k != "evaluate"},
            "seeds": list(SEEDS), "episodes": len(SEEDS),
            "avg_score": round(float(s.mean()), 2),
            "avg_init_score": round(float(b.mean()), 2),
            "avg_gain": round(float(d.mean()), 2),
            "sec_per_episode": round(per, 3),
            "measured_at": now.isoformat(timespec="seconds"),
            "per_seed": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
