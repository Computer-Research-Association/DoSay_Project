"""
알고리즘 에이전트 버전별 실험 결과를 뽑고 비교하는 간단한 도구.

사용법 (프로젝트 루트에서):
    # 한 버전 실행 → results/<버전>.json 저장
    python algorithm/experiment/experiment.py run Greedy_V1a --n-games 100

    # 지금까지 저장된 모든 결과 비교 표 출력
    python algorithm/experiment/experiment.py compare

설계 메모:
- 모든 게임을 고정 시드 세트(base_seed ~ base_seed+n-1)로 돌린다.
  → 버전이 달라도 "같은 보드 시퀀스"를 풀게 되어 공정한 A/B 비교가 된다.
  (executor.reset(seed)가 Board.from_seed(seed)로 시드를 제대로 전달하므로
   옛 simulator의 seed=None 재현성 버그가 여기엔 없다.)
- 버전 하나 = algorithm/models/version/<Type>_<Version>.py 파일 하나.
  그 파일의 heuristic + weight가 곧 실험 설정이므로, 결과 JSON에 그대로 기록한다.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import cast
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # .../DoSay
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))              # executor 내부의 agents.dtos import용
os.chdir(ROOT)                                       # _load_model의 relative_to(cwd) 대응

from algorithm.models.model_executor import GreedyExecutor   # noqa: E402
from algorithm.models.utils import HeuristicRegistry          # noqa: E402

GRID = (9, 18)
TOTAL_CELLS = GRID[0] * GRID[1]
VERSION_DIR = ROOT / "algorithm" / "models" / "version"
RESULTS_DIR = ROOT / "results"

# --- 병렬 실행용 (게임끼리 독립적이라 프로세스로 나눠 돌림) ---------------
_WORKER_EXECUTOR = None   # 프로세스마다 1개만 생성해 재사용

def _worker_init(model_path_str: str):
    """워커 프로세스 시작 시 executor를 한 번만 만든다."""
    global _WORKER_EXECUTOR
    _WORKER_EXECUTOR = GreedyExecutor(GRID, Path(model_path_str))

def _play_one(seed: int):
    """게임 1판 실행 → (score, steps, is_clear). 고정 시드라 결정적."""
    ex = _WORKER_EXECUTOR
    ex.reset(seed)
    steps, score, is_clear = 0, 0, False
    while True:
        is_over, is_clear = ex.board.is_done()
        if is_over:
            break
        score += ex.do_step()
        steps += 1
    return score, steps, is_clear


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def run_model(version: str, n_games: int = 100, base_seed: int = 1234) -> dict:
    model_path = (VERSION_DIR / f"{version}.py").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"버전 파일이 없습니다: {model_path}")

    ex = GreedyExecutor(GRID, model_path)
    registry = cast(HeuristicRegistry, ex.cls.registry)
    features = {e.name: e.weight for e in registry.entries}   # 실험 설정 = 피처+가중치

    seeds = [base_seed + i for i in range(n_games)]
    n_workers = min(os.cpu_count() or 1, n_games)
    t0 = time.perf_counter()

    # 게임끼리 독립적 → 프로세스 풀로 병렬 실행 (고정 시드라 결과는 직렬과 동일)
    with Pool(processes=n_workers, initializer=_worker_init,
              initargs=(str(model_path),)) as pool:
        results_list = pool.map(_play_one, seeds)

    scores = [r[0] for r in results_list]
    turns = [r[1] for r in results_list]
    clear_count = sum(1 for r in results_list if r[2])

    elapsed = time.perf_counter() - t0
    scores_arr = np.array(scores)

    result = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "n_games": n_games,
        "base_seed": base_seed,
        "features": features,
        "metrics": {
            "avg_score": round(float(scores_arr.mean()), 2),
            "std_score": round(float(scores_arr.std()), 2),
            "min_score": int(scores_arr.min()),
            "max_score": int(scores_arr.max()),
            "avg_ratio": round(float(scores_arr.mean()) / TOTAL_CELLS, 3),
            "avg_turns": round(float(np.mean(turns)), 2),
            "clear_rate": round(clear_count / n_games, 3),
            "sec_per_game": round(elapsed / n_games, 3),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{version}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _print_result(r: dict) -> None:
    m = r["metrics"]
    print(f"\n── {r['version']} ─────────────────────────────")
    print(f" avg score  : {m['avg_score']}  (ratio {m['avg_ratio']})")
    print(f" std        : {m['std_score']}")
    print(f" min / max  : {m['min_score']} / {m['max_score']}")
    print(f" avg turns  : {m['avg_turns']}")
    print(f" clear rate : {m['clear_rate']:.1%}")
    print(f" n_games    : {r['n_games']}  (seed {r['base_seed']}, git {r['git_commit']})")
    print(f" features   : {r['features']}")


def compare() -> None:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"저장된 결과가 없습니다. 먼저 'run'으로 실험을 돌리세요. ({RESULTS_DIR})")
        return

    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows.sort(key=lambda r: r["metrics"]["avg_score"], reverse=True)

    print(f"\n{'version':<16}{'avg':>8}{'std':>7}{'ratio':>8}{'clear':>8}{'turns':>8}   features")
    print("─" * 90)
    for r in rows:
        m = r["metrics"]
        feat = ", ".join(f"{k}={v}" for k, v in r["features"].items())
        print(f"{r['version']:<16}{m['avg_score']:>8}{m['std_score']:>7}"
              f"{m['avg_ratio']:>8}{m['clear_rate']*100:>7.1f}%{m['avg_turns']:>8}   {feat[:40]}")
    print(f"\n(avg_score 내림차순 정렬 · {len(rows)}개 버전)")


def main():
    p = argparse.ArgumentParser(description="알고리즘 버전 실험 실행/비교")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="한 버전을 N판 돌려 결과 저장")
    pr.add_argument("version", help="버전 이름 (예: Greedy_V1a) — algorithm/models/version/<이름>.py")
    pr.add_argument("--n-games", type=int, default=100)
    pr.add_argument("--seed", type=int, default=1234, help="고정 시드 시작값")

    sub.add_parser("compare", help="results/ 의 모든 결과 비교 표")

    args = p.parse_args()

    if args.cmd == "run":
        print(f"'{args.version}' {args.n_games}판 실행 중... (고정 시드 {args.seed})")
        r = run_model(args.version, n_games=args.n_games, base_seed=args.seed)
        _print_result(r)
        print(f"\n→ 저장: results/{args.version}.json")
    elif args.cmd == "compare":
        compare()


if __name__ == "__main__":
    main()
