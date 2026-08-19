"""성능 측정.

대화형으로도 쓰고, RunQueue.py 가 무인으로 호출하기도 한다.

    python runs/measure.py                                   # 골라서 측정
    python runs/measure.py --checkpoint <경로> --episodes 100 # 인자로 측정
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import game
from agents.ai.agent import AIAgent
from agents.base import Agent
from agents.dtos import MeasureResult
from agents.selector import agent_class_for, select_agent
from agents.utils import format_dataclass_box
from ai.training import resolve_device
from system_info import format_system_info

ROOT_DIR = Path(game.__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
BENCHMARK_LOG_DIR = ROOT_DIR / "ai" / "logs" / "benchmark"
DEFAULT_EPISODES = 100
BASE_SEED = 1234
ROWS, COLS = 9, 18

os.chdir(ROOT_DIR)


def benchmark(agent: Agent, source: Path, episodes: int = DEFAULT_EPISODES,
              base_seed: int = BASE_SEED, progress_every: int = 0) -> MeasureResult:
    """같은 시드 묶음으로 여러 판을 돌려 결과를 모은다."""
    start = time.time()
    steps_list, score_list = [], []

    for i in range(episodes):
        steps, score = agent.run_episode(board_source=base_seed + i, render=False, delay=0)
        steps_list.append(steps)
        score_list.append(score)
        # 첫 판은 항상 찍는다. 무인 실행에서 '살아 있는가'를 몇 초 안에 알 수 있어야
        # 멈춤을 밤새 방치하지 않는다.
        done = i + 1
        if progress_every and (done == 1 or done % progress_every == 0):
            rate = (time.time() - start) / done
            print(f"  [{done:>4}/{episodes}] 평균 {np.mean(score_list):6.2f}  "
                  f"(판당 {rate:.1f}s, 남은 시간 약 {int(rate * (episodes - done))}s)", flush=True)

    elapsed = time.time() - start
    best = int(np.argmax(score_list))
    worst = int(np.argmin(score_list))
    info = agent.get_info()

    return MeasureResult(
        agent=f"{getattr(info, 'model_name', getattr(info, 'algorithm_name', '?'))}"
              f" / V{info.agent_version}",
        source_path=source,
        episodes=episodes,
        base_seed=base_seed,
        avg_moves=round(float(np.mean(steps_list)), 2),
        avg_score=round(float(np.mean(score_list)), 2),
        std_score=round(float(np.std(score_list)), 2),
        best_score=int(score_list[best]),
        best_seed=base_seed + best,
        worst_score=int(score_list[worst]),
        worst_seed=base_seed + worst,
        elapsed_sec=round(elapsed, 1),
        sec_per_episode=round(elapsed / episodes, 3),
        measured_at=datetime.now().isoformat(timespec="seconds"),
        scores=[int(s) for s in score_list],
    )


def write_histogram(result: MeasureResult) -> Path | None:
    """점수 분포를 텐서보드 히스토그램으로 남긴다.

    폴더를 seed/판수 로 묶는 이유: 히스토그램은 같은 조건에서 잰 것끼리만
    겹쳐 볼 의미가 있다. 같은 상위 폴더 안에 모델별 런이 들어가므로
    텐서보드에서 한 화면에 겹쳐 비교된다.

        ai/logs/benchmark/seed1234_ep100/V9b_DQN_12000000/
        ai/logs/benchmark/seed1234_ep100/V8c_MaskablePPO_2000000/
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("  (torch.utils.tensorboard 가 없어 히스토그램을 건너뜁니다)")
        return None

    group = BENCHMARK_LOG_DIR / f"seed{result.base_seed}_ep{result.episodes}"
    run_dir = group / result.source_path.stem
    scores = np.asarray(result.scores, dtype=np.float64)

    with SummaryWriter(log_dir=str(run_dir)) as writer:
        # step 을 학습 스텝으로 두면 같은 모델의 체크포인트별 분포 변화도 볼 수 있다
        step = _train_steps(result.source_path.stem)
        writer.add_histogram("benchmark/score", scores, global_step=step)
        writer.add_scalar("benchmark/avg_score", result.avg_score, step)
        writer.add_scalar("benchmark/std_score", result.std_score, step)
        writer.add_scalar("benchmark/avg_moves", result.avg_moves, step)
    return run_dir


def _train_steps(stem: str) -> int:
    """파일명 끝의 학습 스텝. 못 읽으면 0."""
    tail = stem.rpartition("_")[2]
    return int(tail) if tail.isdigit() else 0


def result_filename(result: MeasureResult) -> str:
    stem = result.source_path.stem
    stamp = result.measured_at.replace(":", "").replace("-", "")
    seed = "" if result.base_seed == BASE_SEED else f"_seed{result.base_seed}"
    return f"{stem}{seed}_{stamp}.json"


def save_result(result: MeasureResult, agent_info, output: Path | None = None) -> Path:
    """JSON 으로 남긴다. 나중에 버전끼리 비교할 수 있도록 에이전트 정보도 함께."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = output or (RESULT_DIR / result_filename(result))
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.to_dict()
    payload["agent_info"] = {
        f: str(v) for f, v in vars(agent_info).items()
    } if agent_info is not None else {}

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="에이전트 성능 측정")
    parser.add_argument("--checkpoint", type=Path,
                        help="측정할 체크포인트(.zip) 또는 알고리즘 모델(.py). 생략하면 물어본다.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"측정 판 수 (기본 {DEFAULT_EPISODES})")
    parser.add_argument("--output", type=Path, help="결과 JSON 경로 (생략하면 results/ 아래 자동)")
    parser.add_argument("--no-save", action="store_true", help="JSON 저장과 텐서보드 기록을 건너뛴다")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="N판마다 진행 상황 출력 (무인 실행 시 유용)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="추론 장치 (기본 auto). 탐색은 한 수마다 신경망을 수십 번 부른다.")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED,
                        help=f"판 seed 시작값 (기본 {BASE_SEED}). 여러 체크포인트 중 최고를 고른 뒤에는 "
                             "다른 값으로 다시 재야 선택 편향이 빠진 점수가 나온다.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.checkpoint is not None:
        source = args.checkpoint if args.checkpoint.is_absolute() else ROOT_DIR / args.checkpoint
        if not source.exists():
            raise SystemExit(f"측정 대상을 찾을 수 없습니다: {source}")
        agent_cls = agent_class_for(source)
    else:
        agent_cls, source = select_agent(agents_dir=ROOT_DIR)

    # AI 는 탐색 때문에 한 수마다 신경망을 수십 번 부른다. GPU 를 놀리면 안 된다.
    device = resolve_device(args.device)
    extra = {"device": device} if issubclass(agent_cls, AIAgent) else {}

    agent = agent_cls((ROWS, COLS), source, **extra)
    display_source = source.relative_to(ROOT_DIR) if source.is_relative_to(ROOT_DIR) else source
    print(agent.get_info_formatted())
    print(format_system_info(agent))
    print(f"\nRunning {args.episodes} episodes ...\n", flush=True)

    result = benchmark(agent, display_source, args.episodes, base_seed=args.base_seed,
                       progress_every=args.progress_every)
    print()
    print(format_dataclass_box("Summary", result))

    if not args.no_save:
        path = save_result(result, agent.get_info(), args.output)
        print(f"\n결과 저장: {path.relative_to(ROOT_DIR)}")
        run_dir = write_histogram(result)
        if run_dir is not None:
            print(f"텐서보드 : {run_dir.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())