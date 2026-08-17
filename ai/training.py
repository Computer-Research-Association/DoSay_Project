"""버전별 train.py 가 공유하는 학습 설정과 콜백.

TOTAL_TIMESTEP / CHECKPOINT_TIMESTEP 을 코드에 박아두는 대신 커맨드라인에서 받는다.
    python ai/models/version/V6.0/train.py --total-timestep 2000000 --checkpoint-timestep 200000
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback

DEFAULT_TOTAL_TIMESTEP = 1_000_000
DEFAULT_CHECKPOINT_TIMESTEP = 100_000

# 버전 폴더 안에서 모델(.zip)이 모이는 폴더.
# 체크포인트든 최종 저장본이든 구분 없이 전부 여기에 들어간다.
MODELS_DIRNAME = "models"


def default_n_envs() -> int:
    """논리 코어 수 - 1. 한 코어는 학습 프로세스 몫으로 남긴다."""
    return max(1, (os.cpu_count() or 0) - 1)


def _cuda_hint() -> str:
    """CPU 전용 빌드일 때 알려줄 재설치 방법."""
    base = torch.__version__.split("+")[0]
    return (
        f"  설치된 torch : {torch.__version__}  (cuda build: {torch.version.cuda})\n"
        "  '+cpu' 빌드는 GPU를 쓸 수 없습니다. CUDA 빌드로 다시 설치하세요:\n"
        "      pip uninstall -y torch\n"
        f"      pip install torch=={base} --index-url https://download.pytorch.org/whl/cu126\n"
        "  (pip 는 로컬 태그를 무시하므로 uninstall 없이는 재설치되지 않습니다)"
    )


def resolve_device(requested: str) -> str:
    """SB3 는 device='cuda' 인데 CUDA 가 없으면 조용히 CPU 로 돌린다.

    몇 시간을 CPU 로 태우고 나서야 알게 되는 사고라, 여기서 먼저 드러낸다.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 를 요청했지만 사용할 수 없습니다.\n" + _cuda_hint())
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def describe_device(device: str) -> str:
    if device == "cuda":
        return f"GPU 학습: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    if torch.cuda.is_available():
        return "CPU 학습 (CUDA 가 있는데도 --device cpu 로 지정됨)"
    # 한글 콘솔(cp949)에서 인코딩할 수 없는 문자를 출력하면 죽는다. em-dash 금지.
    return "CPU 학습. GPU 를 쓰려면:\n" + _cuda_hint()


@dataclass(frozen=True)
class TrainConfig:
    version_dir: Path
    total_timestep: int
    checkpoint_timestep: int
    n_envs: int
    device: str
    verbose: int = 0

    @property
    def models_dir(self) -> Path:
        return self.version_dir / MODELS_DIRNAME

    @property
    def log_dir(self) -> Path:
        return self.version_dir.parents[2] / "logs"  # ai/models/version/<버전> -> ai/logs

    @property
    def save_freq(self) -> int:
        """콜백 호출 횟수 기준 저장 주기. 호출 1회 = 환경 수만큼의 스텝."""
        return max(self.checkpoint_timestep // self.n_envs, 1)


def model_filename(algorithm: str, version: str, steps: int) -> str:
    """model_loader 가 파싱하는 파일명 규격: V(버전)_(알고리즘)_(학습스텝)"""
    return f"V{version}_{algorithm}_{steps}"


def save_model(model, config: TrainConfig, algorithm: str, version: str, steps: int) -> Path:
    config.models_dir.mkdir(parents=True, exist_ok=True)
    path = config.models_dir / f"{model_filename(algorithm, version, steps)}.zip"
    model.save(str(path))
    return path


def parse_train_args(version_dir: Path) -> TrainConfig:
    n_envs_default = default_n_envs()

    parser = argparse.ArgumentParser(description=f"{version_dir.name} AI 학습")
    parser.add_argument("--total-timestep", type=int, default=DEFAULT_TOTAL_TIMESTEP,
                        help=f"총 학습 스텝 (기본 {DEFAULT_TOTAL_TIMESTEP:,})")
    parser.add_argument("--checkpoint-timestep", type=int, default=DEFAULT_CHECKPOINT_TIMESTEP,
                        help=f"체크포인트 저장 주기 (기본 {DEFAULT_CHECKPOINT_TIMESTEP:,})")
    parser.add_argument("--n-envs", type=int, default=n_envs_default,
                        help="병렬 환경 수 (기본: 논리 코어 수 - 1)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto=있으면 GPU, cuda=없으면 에러로 알림 (기본 auto)")
    parser.add_argument("--verbose", type=int, choices=(0, 1), default=0,
                        help="1이면 SB3 표를 매번 출력한다 (기본 0). 0이어도 체크포인트마다 "
                             "한 줄 찍히고 텐서보드에는 전부 기록된다.")
    args = parser.parse_args()

    if min(args.total_timestep, args.checkpoint_timestep, args.n_envs) <= 0:
        parser.error("스텝 수와 환경 수는 1 이상이어야 합니다.")
    if args.checkpoint_timestep < args.n_envs:
        parser.error(
            f"--checkpoint-timestep({args.checkpoint_timestep}) 이 환경 수({args.n_envs})보다 작습니다. "
            "매 스텝 저장하게 되므로 더 큰 값을 주세요."
        )

    return TrainConfig(
        version_dir=version_dir,
        total_timestep=args.total_timestep,
        checkpoint_timestep=args.checkpoint_timestep,
        n_envs=args.n_envs,
        device=resolve_device(args.device),
        verbose=args.verbose,
    )


class ScoreCallback(BaseCallback):
    """에피소드가 끝날 때의 실제 점수를 텐서보드에 기록한다."""

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info and "score" in info:
                self.logger.record_mean("rollout/ep_score_mean", info["score"])
        return True


class CheckpointSaver(BaseCallback):
    """CHECKPOINT_TIMESTEP 마다 버전 폴더의 models/ 에 저장한다.

    파일명의 스텝 수는 실제 num_timesteps 가 아니라 의도한 저장 지점
    (CHECKPOINT_TIMESTEP 의 배수)이다. 병렬 환경 때문에 실제 스텝은 그 지점을
    조금 넘거나 못 미치는데, 파일명이 100000, 200000 처럼 떨어지는 편이
    measure.py 에서 고르기 쉽다.

    최종 저장본도 같은 규칙(TOTAL_TIMESTEP)을 쓰므로, TOTAL 이 CHECKPOINT 의
    배수면 마지막 체크포인트와 파일명이 겹쳐 덮어쓰인다. 같은 지점의 모델이라
    문제가 없고, 덕분에 '최종본'을 따로 구분해 둘 필요도 없다.
    """

    def __init__(self, config: TrainConfig, algorithm: str, version: str) -> None:
        super().__init__(verbose=1)   # verbose=0 이어도 이 한 줄은 남긴다
        self.config = config
        self.algorithm = algorithm
        self.version = version

    def _on_step(self) -> bool:
        if self.n_calls % self.config.save_freq != 0:
            return True

        milestone = (self.n_calls // self.config.save_freq) * self.config.checkpoint_timestep
        path = save_model(self.model, self.config, self.algorithm, self.version, milestone)
        # SB3 표를 끄면(verbose=0) 화면이 완전히 조용해진다. 살아 있는지 알 수 있도록
        # 체크포인트마다 한 줄만 남긴다.
        print(f"[{self.num_timesteps:>12,} steps] {path.name}", flush=True)
        return True
