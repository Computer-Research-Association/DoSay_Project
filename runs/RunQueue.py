"""학습/벤치마크 작업을 큐에 쌓아 무인으로 돌린다.

용도가 '자기 전에 걸어두고 아침에 결과 보기' 라서 두 가지를 원칙으로 잡았다.

  1. 물어보는 것은 전부 시작 전에 끝낸다. 실행이 시작되면 입력을 요구하지 않는다.
  2. 한 작업이 실패해도 큐는 계속 간다. 결과는 매 작업이 끝날 때마다 기록하므로,
     중간에 죽어도 그때까지의 결과는 남는다.

학습과 벤치마크 모두 **별도 프로세스**로 띄운다. 버전마다 env.py / model.py 를
같은 이름으로 import 하므로 한 프로세스에서 여러 버전을 다룰 수 없고, 프로세스를
분리해 두면 한 작업이 죽어도 큐가 함께 죽지 않는다.

    python runs/RunQueue.py
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import game
from agents.ai.model_loader import describe_checkpoint, find_checkpoints
from agents.utils import format_box, print_box, prompt_int, prompt_yes_no, select_from
from ai.training import (
    DEFAULT_CHECKPOINT_TIMESTEP, DEFAULT_TOTAL_TIMESTEP, default_n_envs, resolve_device,
)
from train import find_versions

ROOT_DIR = Path(game.__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent
RESULT_DIR = ROOT_DIR / "results"
DEFAULT_EPISODES = 100

os.chdir(ROOT_DIR)


@dataclass
class Settings:
    device: str
    n_envs: int
    episodes: int


@dataclass
class Job:
    kind: str                        # "train" | "measure"
    version_dir: Path | None = None
    checkpoint: Path | None = None
    total_timestep: int = 0
    checkpoint_timestep: int = 0
    measure_after: bool = False

    def describe(self) -> str:
        if self.kind == "train":
            target = f"{self.version_dir.parent.name}/{self.version_dir.name}"  # type: ignore[union-attr]
            tail = " + 벤치마크" if self.measure_after else ""
            return f"학습  {target}  {self.total_timestep:,} steps" f" (체크포인트 {self.checkpoint_timestep:,}){tail}"
        return f"벤치마크  {self.checkpoint.relative_to(ROOT_DIR)}"  # type: ignore[union-attr]


@dataclass
class JobRecord:
    job: str
    status: str = "pending"
    started_at: str = ""
    elapsed_sec: float = 0.0
    detail: str = ""
    avg_score: float | None = None
    result_file: str | None = None

    def to_dict(self) -> dict:
        data = dict(vars(self))
        return data


# ── 큐 구성 (여기서 물어볼 것은 다 물어본다) ──────────────────────────────

def ask_settings() -> Settings:
    device_labels = ["auto (있으면 GPU, 없으면 CPU)", "cuda (없으면 즉시 중단)", "cpu"]
    device = ("auto", "cuda", "cpu")[select_from("Device", device_labels)]
    resolved = resolve_device(device)   # cuda 를 못 쓰면 큐를 짜기 전에 알려준다

    n_envs = prompt_int("병렬 환경 수 (n-envs)", default_n_envs())
    episodes = prompt_int("벤치마크 판 수", DEFAULT_EPISODES)
    return Settings(device=resolved, n_envs=n_envs, episodes=episodes)


def add_training_job(queue: list[Job]) -> bool:
    """버전을 하나 골라 학습 작업을 추가한다. 큐 실행을 고르면 False."""
    versions = find_versions()
    if not versions:
        raise SystemExit(f"{ROOT_DIR / 'ai' / 'models'} 아래에 train.py 를 가진 버전이 없습니다.")

    labels = [f"학습: {d.parent.name}/{d.name}" for d in versions]
    labels.append("벤치마크만 추가 (이미 학습된 모델)")
    labels.append("── 추가 그만하고 큐 실행 ──")

    choice = select_from(f"Add Job  (현재 {len(queue)}개)", labels)

    if choice == len(labels) - 1:
        return False

    if choice == len(labels) - 2:
        add_measure_job(queue)
        return True

    version_dir = versions[choice]
    total = prompt_int("  TOTAL_TIMESTEP", DEFAULT_TOTAL_TIMESTEP)
    checkpoint = prompt_int("  CHECKPOINT_TIMESTEP", DEFAULT_CHECKPOINT_TIMESTEP)
    measure_after = prompt_yes_no("  학습이 끝나면 벤치마크도 돌릴까요?", default=True)

    queue.append(Job(kind="train", version_dir=version_dir, total_timestep=total,
                     checkpoint_timestep=checkpoint, measure_after=measure_after))
    return True


def add_measure_job(queue: list[Job]) -> None:
    """체크포인트 하나, 또는 한 버전의 전체 체크포인트를 큐에 넣는다.

    '전체' 가 있는 이유: 길게 학습해 두고 나중에 제일 좋은 지점을 고르려면
    체크포인트를 전부 재봐야 하는데, 탐색이 없는 모델은 100판에 30초라
    16개를 다 재도 8분이면 끝난다. 하나씩 고르게 두면 그게 더 고생이다.
    """
    trained = [d for d in find_versions() if find_checkpoints(d)]
    if not trained:
        print("  (아직 학습된 체크포인트가 없습니다)")
        return

    labels = [f"{d.parent.name}/{d.name}  (체크포인트 {len(find_checkpoints(d))}개)" for d in trained]
    version_dir = trained[select_from("Select Version", labels)]
    checkpoints = find_checkpoints(version_dir)

    if len(checkpoints) > 1:
        options = [f"── 전체 {len(checkpoints)}개 다 재기 ──"]
        options += [describe_checkpoint(c).strip() for c in checkpoints]
        choice = select_from(f"Select Checkpoint ({version_dir.name})", options)
        if choice == 0:
            queue.extend(Job(kind="measure", checkpoint=c) for c in checkpoints)
            return
        checkpoints = [checkpoints[choice - 1]]

    queue.append(Job(kind="measure", checkpoint=checkpoints[0]))


def build_queue() -> tuple[Settings, list[Job]]:
    settings = ask_settings()
    queue: list[Job] = []
    while add_training_job(queue):
        pass
    return settings, queue


# ── 실행 ─────────────────────────────────────────────────────────────────

def trained_checkpoint(version_dir: Path, total: int) -> Path | None:
    """학습이 끝난 뒤 벤치마크할 대상. 없으면 None."""
    checkpoints = find_checkpoints(version_dir)
    if not checkpoints:
        return None
    exact = [c for c in checkpoints if c.stem.endswith(f"_{total}")]
    return exact[-1] if exact else checkpoints[-1]


def run_subprocess(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    sys.stdout.flush()

    proc = subprocess.Popen(cmd, cwd=ROOT_DIR)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.kill()          # Ctrl+C 시 자식을 남기지 않는다
        proc.wait()
        raise


def run_training(job: Job, settings: Settings) -> int:
    return run_subprocess([
        sys.executable, str(job.version_dir / "train.py"),  # type: ignore[operator]
        "--total-timestep", str(job.total_timestep),
        "--checkpoint-timestep", str(job.checkpoint_timestep),
        "--n-envs", str(settings.n_envs),
        "--device", settings.device,
    ])


def run_measure(checkpoint: Path, settings: Settings, record: JobRecord) -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = RESULT_DIR / f"{checkpoint.stem}_{stamp}.json"

    code = run_subprocess([
        sys.executable, str(RUNS_DIR / "measure.py"),
        "--checkpoint", str(checkpoint),
        "--episodes", str(settings.episodes),
        "--output", str(output),
        "--progress-every", str(max(settings.episodes // 20, 1)),
        "--device", settings.device,
    ])

    if code == 0 and output.exists():
        record.result_file = str(output.relative_to(ROOT_DIR))
        record.avg_score = json.loads(output.read_text(encoding="utf-8")).get("avg_score")
    return code


def write_log(log_path: Path, settings: Settings, records: list[JobRecord]) -> None:
    """매 작업이 끝날 때마다 덮어쓴다. 중간에 죽어도 여기까지는 남는다."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": records[0].started_at if records else "",
        "settings": vars(settings),
        "jobs": [r.to_dict() for r in records],
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_queue(settings: Settings, queue: list[Job]) -> list[JobRecord]:
    records = [JobRecord(job=job.describe()) for job in queue]
    log_path = RESULT_DIR / f"queue_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    started = time.time()

    for index, (job, record) in enumerate(zip(queue, records), start=1):
        record.started_at = datetime.now().isoformat(timespec="seconds")
        job_start = time.time()
        print("\n" + "=" * 72)
        print(f"[{index}/{len(queue)}]  {job.describe()}")
        print(f"          시작 {record.started_at}   (누적 {timedelta(seconds=int(time.time() - started))})")
        print("=" * 72, flush=True)

        try:
            if job.kind == "train":
                code = run_training(job, settings)
                if code != 0:
                    record.status, record.detail = "failed", f"학습이 코드 {code} 로 종료"
                else:
                    record.status = "done"
                    if job.measure_after:
                        target = trained_checkpoint(job.version_dir, job.total_timestep)  # type: ignore[arg-type]
                        if target is None:
                            record.detail = "학습은 끝났지만 체크포인트를 찾지 못해 벤치마크를 건너뜀"
                        elif run_measure(target, settings, record) != 0:
                            record.detail = "학습은 끝났지만 벤치마크가 실패"
            else:
                code = run_measure(job.checkpoint, settings, record)  # type: ignore[arg-type]
                record.status = "done" if code == 0 else "failed"
                if code != 0:
                    record.detail = f"벤치마크가 코드 {code} 로 종료"

        except KeyboardInterrupt:
            record.status = "interrupted"
            record.elapsed_sec = round(time.time() - job_start, 1)
            write_log(log_path, settings, records)
            print("\n중단되었습니다. 남은 작업은 실행하지 않습니다.")
            print(f"기록: {log_path.relative_to(ROOT_DIR)}")
            return records
        except Exception as exc:                      # 한 작업의 사고가 큐를 죽이지 않게
            record.status, record.detail = "error", f"{type(exc).__name__}: {exc}"

        record.elapsed_sec = round(time.time() - job_start, 1)
        write_log(log_path, settings, records)

    print(f"\n전체 소요 {timedelta(seconds=int(time.time() - started))}")
    print(f"기록: {log_path.relative_to(ROOT_DIR)}")
    return records


def summarize(records: list[JobRecord]) -> None:
    lines = []
    for index, record in enumerate(records, start=1):
        score = f"{record.avg_score:6.2f}" if record.avg_score is not None else "   -  "
        elapsed = str(timedelta(seconds=int(record.elapsed_sec)))
        lines.append(f"[{index}] {record.status:<11} avg={score}  {elapsed:>8}  {record.job}")
        if record.detail:
            lines.append(f"     {record.detail}")
    print()
    print(format_box("Queue Result", lines or ["(작업 없음)"]))


def main() -> int:
    settings, queue = build_queue()
    if not queue:
        print("큐가 비어 있습니다.")
        return 0

    print()
    print_box("Queue", [f"device {settings.device} / n-envs {settings.n_envs} "
                        f"/ 벤치마크 {settings.episodes}판"]
                       + [f"[{i}] {job.describe()}" for i, job in enumerate(queue, 1)])

    if not prompt_yes_no("\n이대로 실행할까요?", default=True):
        print("취소했습니다.")
        return 0

    summarize(run_queue(settings, queue))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
