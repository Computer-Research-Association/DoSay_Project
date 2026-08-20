from pathlib import Path
from typing import Callable, Tuple, Type

from agents.base import Agent
from agents.ai.agent import AIAgent
from agents.ai.model_loader import describe_checkpoint, find_checkpoints
from agents.algorithm.agent import AlgoAgent
from agents.utils import select_from

AI_MODEL_DIR = "ai/models"          # ai/models/<알고리즘>/<버전>/
ALGO_VERSION_GLOB = "algorithm/models/version/*.py"


def _trained_versions(algorithm_dir: Path) -> list[Path]:
    return [d for d in sorted(algorithm_dir.iterdir()) if d.is_dir() and find_checkpoints(d)]


def _select_ai(root: Path) -> Path:
    """알고리즘 -> 버전 -> 학습 스텝 순으로 좁혀 체크포인트 하나를 고른다."""
    base = root / AI_MODEL_DIR
    algorithms = sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
    trained = [d for d in algorithms if _trained_versions(d)]  # 학습 전 알고리즘은 빠진다

    if not trained:
        raise RuntimeError(
            "학습된 AI 모델이 없습니다.\n"
            f"{base} 아래 버전을 골라 'python runs/train.py' 로 먼저 학습해 주세요."
        )

    algorithm_dir = trained[select_from("Select Algorithm", [d.name for d in trained])]

    versions = _trained_versions(algorithm_dir)
    version_dir = versions[select_from(f"Select Version ({algorithm_dir.name})",
                                       [d.name for d in versions])]

    checkpoints = find_checkpoints(version_dir)
    if len(checkpoints) == 1:
        return checkpoints[0]  # 고를 것이 없으면 묻지 않는다

    labels = [describe_checkpoint(c) for c in checkpoints]
    return checkpoints[select_from(f"Select Timestep ({version_dir.name})", labels)]


def _select_algorithm(root: Path) -> Path:
    """종류 -> 버전 순으로 좁혀 모델 파일 하나를 고른다.

    AI 쪽과 같은 모양으로 두 단계다. 버전 파일이 60개를 넘어서 한 목록에 늘어놓으면
    화면을 넘겨 가며 골라야 한다. 종류는 파일명 앞부분(Greedy/Beam/Anneal)이다.
    """
    candidates = sorted(root.glob(ALGO_VERSION_GLOB))
    if not candidates:
        raise RuntimeError(f"{root / ALGO_VERSION_GLOB} 에 사용 가능한 알고리즘이 없습니다.")

    by_type: dict[str, list[Path]] = {}
    for path in candidates:
        by_type.setdefault(path.stem.split("_")[0], []).append(path)

    types = sorted(by_type)
    chosen = types[0] if len(types) == 1 else types[
        select_from("Select ALGORITHM Type",
                    [f"{name}  ({len(by_type[name])}개)" for name in types])]

    versions = by_type[chosen]
    if len(versions) == 1:
        return versions[0]  # 고를 것이 없으면 묻지 않는다
    return versions[select_from(f"Select Version ({chosen})", [p.stem for p in versions])]


AGENT_REGISTRY: dict[str, tuple[Type[Agent], Callable[[Path], Path]]] = {
    "ai": (AIAgent, _select_ai),
    "algorithm": (AlgoAgent, _select_algorithm),
}


def agent_class_for(source: Path) -> Type[Agent]:
    """경로만 보고 에이전트 종류를 정한다 (대화형 선택 없이 쓰는 경로용)."""
    if source.suffix == ".zip":
        return AIAgent
    if source.suffix == ".py":
        return AlgoAgent
    raise ValueError(
        f"에이전트 종류를 알 수 없습니다: {source}\n"
        "AI 는 체크포인트(.zip), algorithm 은 모델 파일(.py) 이어야 합니다."
    )


def select_agent(agents_dir: Path) -> Tuple[Type[Agent], Path]:
    types = list(AGENT_REGISTRY.keys())
    agent_type = types[select_from("Select Agent Type", types)]

    agent_cls, select_source = AGENT_REGISTRY[agent_type]
    source = select_source(agents_dir)
    print()

    return (agent_cls, source)
