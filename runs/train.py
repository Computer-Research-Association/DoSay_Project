"""AI 학습 실행기 (AI 전용 — algorithm 은 학습 대상이 아니다).

버전 폴더(ai/models/version/*)를 골라 그 안의 train.py 를 실행한다.
버전마다 env.py / model.py 를 같은 이름으로 import 하므로 한 프로세스에서
여러 버전을 다룰 수 없다. 그래서 import 대신 서브프로세스로 띄운다.

학습 로그는 ai/logs 에 쌓기만 한다. 보는 건 TensorBoard 쪽 몫이다.

    python runs/train.py                        # 대화형으로 선택
    python runs/train.py --version V6.0 --total-timestep 2000000
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import game
from agents.dtos import TrainInfo
from agents.utils import format_dataclass_box, prompt_int, select_from
from ai.training import (
    DEFAULT_CHECKPOINT_TIMESTEP, DEFAULT_TOTAL_TIMESTEP, MODELS_DIRNAME, resolve_device,
)

ROOT_DIR = Path(game.__file__).resolve().parent.parent
MODEL_ROOT = ROOT_DIR / "ai" / "models"   # ai/models/<알고리즘>/<버전>/

os.chdir(ROOT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 모델 학습 실행기")
    parser.add_argument("--version", help="학습할 버전 (예: V8a 또는 MaskablePPO/V8a). 생략하면 물어본다.")
    parser.add_argument("--total-timestep", type=int, help=f"총 학습 스텝 (기본 {DEFAULT_TOTAL_TIMESTEP:,})")
    parser.add_argument("--checkpoint-timestep", type=int, help=f"체크포인트 주기 (기본 {DEFAULT_CHECKPOINT_TIMESTEP:,})")
    parser.add_argument("--n-envs", type=int, help="병렬 환경 수 (기본: 논리 코어 수 - 1)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto=있으면 GPU, cuda=없으면 에러로 알림 (기본 auto)")
    return parser.parse_args()


def find_versions(algorithm: str | None = None) -> list[Path]:
    """train.py 를 가진 버전 폴더. 아직 학습 전인 버전도 당연히 포함된다."""
    if not MODEL_ROOT.is_dir():
        return []
    pattern = f"{algorithm}/*/train.py" if algorithm else "*/*/train.py"
    return sorted(p.parent for p in MODEL_ROOT.glob(pattern))


def select_version(name: str | None) -> Path:
    versions = find_versions()
    if not versions:
        raise SystemExit(f"{MODEL_ROOT} 아래에 train.py 를 가진 버전 폴더가 없습니다.")

    if name is not None:
        # "V8a" 또는 "MaskablePPO/V8a" 둘 다 받는다
        wanted = name.replace("\\", "/").lower()
        for version_dir in versions:
            full = f"{version_dir.parent.name}/{version_dir.name}".lower()
            if wanted in (version_dir.name.lower(), full):
                return version_dir
        available = ", ".join(f"{d.parent.name}/{d.name}" for d in versions)
        raise SystemExit(f"'{name}' 버전을 찾을 수 없습니다. (가능: {available})")

    algorithms = sorted({d.parent for d in versions})
    algorithm_dir = algorithms[select_from("Select Algorithm", [d.name for d in algorithms])]

    candidates = [d for d in versions if d.parent == algorithm_dir]
    return candidates[select_from(f"Select Version to Train ({algorithm_dir.name})",
                                  [d.name for d in candidates])]


def resolve_timesteps(args: argparse.Namespace) -> tuple[int, int]:
    total = args.total_timestep
    if total is None:
        total = prompt_int("TOTAL_TIMESTEP", DEFAULT_TOTAL_TIMESTEP)

    checkpoint = args.checkpoint_timestep
    if checkpoint is None:
        checkpoint = prompt_int("CHECKPOINT_TIMESTEP", DEFAULT_CHECKPOINT_TIMESTEP)

    return total, checkpoint


def run_training(version_dir: Path, total: int, checkpoint: int,
                 n_envs: int | None, device: str) -> int:
    cmd = [
        sys.executable, str(version_dir / "train.py"),
        "--total-timestep", str(total),
        "--checkpoint-timestep", str(checkpoint),
        "--device", device,
    ]
    if n_envs is not None:
        cmd += ["--n-envs", str(n_envs)]

    sys.stdout.flush()  # 자식 프로세스 출력과 순서가 뒤섞이지 않게
    return subprocess.call(cmd, cwd=ROOT_DIR)


def main() -> int:
    args = parse_args()

    version_dir = select_version(args.version)
    total, checkpoint = resolve_timesteps(args)
    device = resolve_device(args.device)  # CUDA 를 못 쓰면 여기서 바로 알려준다

    info = TrainInfo(
        agent_version=f"{version_dir.parent.name} / {version_dir.name}",
        total_timestep=total,
        checkpoint_timestep=checkpoint,
        save_path=(version_dir / MODELS_DIRNAME).relative_to(ROOT_DIR),
        device=device,
        n_envs=args.n_envs,
    )
    print()
    print(format_dataclass_box("Training Info", info))
    print()

    code = run_training(version_dir, total, checkpoint, args.n_envs, device)
    if code != 0:
        print(f"\n학습이 코드 {code} 로 종료되었습니다.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
